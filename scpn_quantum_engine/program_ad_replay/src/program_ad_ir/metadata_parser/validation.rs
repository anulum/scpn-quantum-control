// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — admitted Program AD metadata parsing

use std::fmt;
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;
use super::Context;

struct BorrowRaw;

impl<'de> DeserializeSeed<'de> for BorrowRaw {
    type Value = &'de RawValue;
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        <&RawValue as serde::Deserialize>::deserialize(deserializer)
    }
}

struct DiscardString;

impl<'de> DeserializeSeed<'de> for DiscardString {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        impl Visitor<'_> for DiscardString {
            type Value = ();
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON string")
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<(), E> { Ok(()) }
        }
        deserializer.deserialize_str(self)
    }
}

struct Children<'a>(&'a Context, bool);

impl<'de> DeserializeSeed<'de> for Children<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        impl<'de> Visitor<'de> for Children<'_> {
            type Value = ();
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON container")
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut sequence: S) -> Result<(), S::Error> {
                while let Some(raw) = sequence.next_element::<&RawValue>()? {
                    validate_raw(self.0, raw).map_err(<S::Error as serde::de::Error>::custom)?;
                }
                Ok(())
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
                while map.next_key_seed(DiscardString)?.is_some() {
                    let raw = map.next_value::<&RawValue>()?;
                    validate_raw(self.0, raw).map_err(<M::Error as serde::de::Error>::custom)?;
                }
                Ok(())
            }
        }
        if self.1 { deserializer.deserialize_map(self) }
        else { deserializer.deserialize_seq(self) }
    }
}

fn validate_raw(context: &Context, raw: &RawValue) -> Result<(), String> {
    context.checkpoint()?;
    let token = raw.get();
    match token.as_bytes().first().copied() {
        Some(b'{') => context.decode(token, Children(context, true)),
        Some(b'[') => context.decode(token, Children(context, false)),
        Some(b'"') => context.decode(token, DiscardString),
        Some(b'-' | b'0'..=b'9') => {
            // core's decimal-to-f64 conversion has no heap allocator. RawValue
            // has already checked JSON syntax; only finite range remains here.
            let number = token.parse::<f64>().map_err(|_| "JSON number is invalid".to_owned())?;
            if number.is_finite() { Ok(()) }
            else { Err("JSON number is out of range".to_owned()) }
        }
        _ => Ok(()),
    }
}

fn validate_depth(context: &Context, source: &str) -> Result<(), String> {
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        if index % 256 == 0 { context.checkpoint()?; }
        if quoted {
            if escaped { escaped = false; }
            else if byte == b'\\' { escaped = true; }
            else if byte == b'"' { quoted = false; }
        } else {
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    depth += 1;
                    // serde_json's default 128 counter refuses at zero after
                    // decrementing for each container, permitting depth 127.
                    if depth >= 128 { return Err("JSON recursion limit exceeded".to_owned()); }
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) fn validate(context: &Context, serialization: &str) -> Result<(), String> {
    let result = (|| {
        validate_depth(context, serialization)?;
        let raw = context.decode(serialization, BorrowRaw)?;
        validate_raw(context, raw)
    })();
    if let Some(error) = context.failure.borrow().as_ref() {
        return Err(error.clone());
    }
    result.map_err(|error: String| format!("program AD IR serialization is invalid JSON: {error}"))
}
