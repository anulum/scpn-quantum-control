// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — admitted Program AD metadata parsing

//! Borrowed JSON fields, fallible owned IR storage, and separate admission.

mod records;
mod validation;

use super::ProgramADEffectIR;
use crate::program_ad_lifecycle::{admit_replay_metadata, replay_checkpoint};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;
use std::{cell::RefCell, fmt, mem::size_of};

struct Context {
    failure: RefCell<Option<String>>,
}

impl Context {
    fn checkpoint(&self) -> Result<(), String> {
        let result = replay_checkpoint();
        if let Err(error) = &result {
            if self.failure.borrow().is_none() {
                self.failure.replace(Some(error.clone()));
            }
        }
        result
    }

    fn charge(&self, bytes: usize) -> Result<(), String> {
        let result = admit_replay_metadata(bytes);
        if let Err(error) = &result {
            if self.failure.borrow().is_none() {
                self.failure.replace(Some(error.clone()));
            }
        }
        result
    }

    fn bytes(&self, count: usize, width: usize) -> Result<usize, String> {
        count
            .checked_mul(width)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| "Program AD parser metadata exceeds native addressability".to_owned())
    }

    fn decode<'de, T>(
        &self,
        raw: &'de str,
        seed: impl DeserializeSeed<'de, Value = T>,
    ) -> Result<T, String> {
        // serde_json's shared byte scratch grows geometrically (minimum eight).
        // Retain a conservative old/new overlap bound for every decoding pass.
        let scratch = self
            .bytes(raw.len(), 3)?
            .checked_add(16)
            .ok_or_else(|| "Program AD parser metadata exceeds native addressability".to_owned())?;
        self.charge(scratch)?;
        let mut deserializer = serde_json::Deserializer::from_str(raw);
        let result = seed.deserialize(&mut deserializer).and_then(|value| {
            deserializer.end()?;
            Ok(value)
        });
        if let Some(error) = self.failure.borrow().as_ref() {
            return Err(error.clone());
        }
        result
            .map_err(|error| format!("program AD IR serialization does not match schema: {error}"))
    }
}

struct FieldKey<'a>(&'a [&'a str]);

impl<'de> DeserializeSeed<'de> for FieldKey<'_> {
    type Value = usize;
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
        deserializer.deserialize_str(self)
    }
}

struct FieldSlots<'a, const N: usize>(&'a Context, &'a [&'a str; N], Option<usize>);

impl<'de, const N: usize> DeserializeSeed<'de> for FieldSlots<'_, N> {
    type Value = [Option<&'de RawValue>; N];
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

struct OwnedString<'a>(&'a Context);

impl<'de> DeserializeSeed<'de> for OwnedString<'_> {
    type Value = String;
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<String, D::Error> {
        deserializer.deserialize_str(self)
    }
}

struct Elements<'a, T> {
    context: &'a Context,
    decode: fn(&Context, &RawValue) -> Result<T, String>,
}

impl<'de, T> DeserializeSeed<'de> for Elements<'_, T> {
    type Value = Vec<T>;
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Vec<T>, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

fn required<'a>(raw: Option<&'a RawValue>, name: &str) -> Result<&'a RawValue, String> {
    raw.ok_or_else(|| {
        format!("program AD IR serialization does not match schema: missing field `{name}`")
    })
}

fn fields<'de, const N: usize>(
    context: &Context,
    raw: &'de RawValue,
    names: &[&str; N],
) -> Result<[Option<&'de RawValue>; N], String> {
    if !matches!(raw.get().as_bytes().first(), Some(b'{' | b'[')) {
        return Err(
            "program AD IR serialization does not match schema: expected metadata record"
                .to_owned(),
        );
    }
    let minimum = if names.last() == Some(&"operation") {
        N - 1
    } else {
        N
    };
    context.decode(raw.get(), FieldSlots(context, names, Some(minimum)))
}

fn string(context: &Context, raw: &RawValue) -> Result<String, String> {
    if !raw.get().starts_with('"') {
        return Err(
            "program AD IR serialization does not match schema: expected string".to_owned(),
        );
    }
    context.decode(raw.get(), OwnedString(context))
}

trait Primitive: Sized {
    fn from_token(token: &str) -> Option<Self>;
}

impl Primitive for usize {
    fn from_token(token: &str) -> Option<Self> {
        token.parse().ok()
    }
}

impl Primitive for bool {
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }
}

fn scalar<T: Primitive>(context: &Context, raw: &RawValue) -> Result<T, String> {
    context.checkpoint()?;
    T::from_token(raw.get()).ok_or_else(|| {
        "program AD IR serialization does not match schema: invalid primitive token".to_owned()
    })
}

fn optional<T>(
    context: &Context,
    raw: Option<&RawValue>,
    decode: fn(&Context, &RawValue) -> Result<T, String>,
) -> Result<Option<T>, String> {
    match raw {
        None => Ok(None),
        Some(raw) if raw.get() == "null" => Ok(None),
        Some(raw) => decode(context, raw).map(Some),
    }
}

fn vector<T>(
    context: &Context,
    raw: &RawValue,
    decode: fn(&Context, &RawValue) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    if !raw.get().starts_with('[') {
        return Err(
            "program AD IR serialization does not match schema: expected metadata list".to_owned(),
        );
    }
    context.decode(raw.get(), Elements { context, decode })
}

pub(super) fn parse(serialization: &str) -> Result<ProgramADEffectIR, String> {
    let context = Context {
        failure: RefCell::new(None),
    };
    validation::validate(&context, serialization)?;
    replay_checkpoint()?;
    records::parse(&context, serialization)
}

impl Visitor<'_> for FieldKey<'_> {
    type Value = usize;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a metadata field name")
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<usize, E> {
        Ok(self
            .0
            .iter()
            .position(|field| *field == value)
            .unwrap_or(usize::MAX))
    }
}

impl<'de, const N: usize> Visitor<'de> for FieldSlots<'_, N> {
    type Value = [Option<&'de RawValue>; N];
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a metadata object")
    }
    fn visit_seq<S: SeqAccess<'de>>(
        self,
        mut sequence: S,
    ) -> Result<Self::Value, S::Error> {
        let minimum = self.2.ok_or_else(|| {
            <S::Error as serde::de::Error>::custom("expected metadata object")
        })?;
        let mut fields = [None; N];
        let mut count = 0;
        while let Some(raw) = sequence.next_element::<&RawValue>()? {
            self.0
                .checkpoint()
                .map_err(<S::Error as serde::de::Error>::custom)?;
            if count == N {
                return Err(<S::Error as serde::de::Error>::custom(
                    "metadata record has excess fields",
                ));
            }
            fields[count] = Some(raw);
            count += 1;
        }
        if count < minimum {
            return Err(<S::Error as serde::de::Error>::custom(
                "metadata record has missing fields",
            ));
        }
        Ok(fields)
    }
    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let mut fields = [None; N];
        while let Some(index) = map.next_key_seed(FieldKey(self.1))? {
            self.0
                .checkpoint()
                .map_err(<M::Error as serde::de::Error>::custom)?;
            let value = map.next_value::<&RawValue>()?;
            if index < N {
                fields[index] = Some(value);
            }
        }
        Ok(fields)
    }
}

impl Visitor<'_> for OwnedString<'_> {
    type Value = String;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string")
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<String, E> {
        self.0.charge(value.len()).map_err(E::custom)?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(value.len())
            .map_err(|_| E::custom("Program AD parser string allocation failed"))?;
        owned.push_str(value);
        Ok(owned)
    }
}

impl<'de, T> Visitor<'de> for Elements<'_, T> {
    type Value = Vec<T>;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a metadata list")
    }
    fn visit_seq<S: SeqAccess<'de>>(self, mut sequence: S) -> Result<Vec<T>, S::Error> {
        let mut values = Vec::new();
        while let Some(raw) = sequence.next_element::<&RawValue>()? {
            self.context
                .checkpoint()
                .map_err(<S::Error as serde::de::Error>::custom)?;
            if values.len() == values.capacity() {
                let capacity = values
                    .capacity()
                    .checked_mul(2)
                    .map(|n| n.max(4))
                    .ok_or_else(|| {
                        <S::Error as serde::de::Error>::custom(
                            "Program AD parser vector capacity overflow",
                        )
                    })?;
                let bytes = self
                    .context
                    .bytes(capacity, size_of::<T>())
                    .map_err(<S::Error as serde::de::Error>::custom)?;
                self.context
                    .charge(bytes)
                    .map_err(<S::Error as serde::de::Error>::custom)?;
                values
                    .try_reserve_exact(capacity - values.len())
                    .map_err(|_| {
                        <S::Error as serde::de::Error>::custom(
                            "Program AD parser vector allocation failed",
                        )
                    })?;
            }
            values.push(
                (self.decode)(self.context, raw)
                    .map_err(<S::Error as serde::de::Error>::custom)?,
            );
        }
        Ok(values)
    }
}
