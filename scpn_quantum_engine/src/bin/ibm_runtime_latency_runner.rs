// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// © Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

use anyhow::{Context, Result, bail};
use clap::Parser;
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::{Duration, Instant};

const DEFAULT_RUNTIME_URL: &str = "https://api.quantum-computing.ibm.com/runtime";
const IBM_API_VERSION: &str = "2025-05-01";

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    payload_matrix: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = DEFAULT_RUNTIME_URL.to_string())]
    runtime_url: String,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    service_crn: Option<String>,
    #[arg(long, default_value_t = 1800)]
    timeout_s: u64,
    #[arg(long, default_value_t = 2.0)]
    poll_interval_s: f64,
}

#[derive(Debug, Deserialize)]
struct PayloadRow {
    lane: String,
    scenario: String,
    payload: Value,
}

#[derive(Debug, Deserialize)]
struct PayloadMatrix {
    schema: String,
    backend: String,
    rows: Vec<PayloadRow>,
}

#[derive(Debug, Serialize)]
struct RunRow {
    lane: String,
    scenario: String,
    job_id: String,
    submit_overhead_seconds: f64,
    submit_to_done_seconds: f64,
    final_status: String,
}

#[derive(Debug, Serialize)]
struct RunReport {
    schema: String,
    backend: String,
    runner: String,
    rows: Vec<RunRow>,
}

#[derive(Debug, Deserialize)]
struct IamResponse {
    access_token: String,
}

fn fetch_iam_token(api_key: &str, client: &Client) -> Result<String> {
    let params = [
        ("grant_type", "urn:ibm:params:oauth:grant-type:apikey"),
        ("apikey", api_key),
    ];
    let response = client
        .post("https://iam.cloud.ibm.com/identity/token")
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .form(&params)
        .send()
        .context("failed to request IAM token")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        bail!("IAM token request failed: {} {}", status, body);
    }
    let iam: IamResponse = response.json().context("invalid IAM token response")?;
    Ok(iam.access_token)
}

fn build_headers(token: &str, service_crn: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("User-Agent", HeaderValue::from_static("python-requests/2.32.5"));
    headers.insert("IBM-API-Version", HeaderValue::from_static(IBM_API_VERSION));
    headers.insert(
        "Service-CRN",
        HeaderValue::from_str(service_crn).context("invalid Service-CRN header value")?,
    );
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {token}"))
            .context("invalid Authorization header value")?,
    );
    Ok(headers)
}

fn submit_job(client: &Client, base_url: &str, headers: &HeaderMap, payload: &Value) -> Result<String> {
    let response = client
        .post(format!("{}/jobs", base_url.trim_end_matches('/')))
        .headers(headers.clone())
        .json(payload)
        .send()
        .context("failed to submit runtime job")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        bail!("job submission failed: {} {}", status, body);
    }
    let body: Value = response.json().context("invalid job submission response")?;
    let job_id = body
        .get("id")
        .and_then(Value::as_str)
        .context("job submission response missing id")?;
    Ok(job_id.to_string())
}

fn poll_status(
    client: &Client,
    base_url: &str,
    headers: &HeaderMap,
    job_id: &str,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<String> {
    let started = Instant::now();
    loop {
        if started.elapsed() > timeout {
            bail!("job {job_id} timed out while waiting for completion");
        }
        let response = client
            .get(format!(
                "{}/jobs/{}?exclude_params=true",
                base_url.trim_end_matches('/'),
                job_id
            ))
            .headers(headers.clone())
            .send()
            .with_context(|| format!("failed to poll job status for {job_id}"))?;
        let status_code = response.status();
        if !status_code.is_success() {
            let body = response.text().unwrap_or_default();
            bail!("job poll failed for {job_id}: {} {}", status_code, body);
        }
        let body: Value = response.json().context("invalid job status response")?;
        let status_value = body
            .get("state")
            .and_then(|state| state.get("status"))
            .and_then(Value::as_str)
            .or_else(|| body.get("status").and_then(Value::as_str))
            .unwrap_or("UNKNOWN");
        let status_upper = status_value.to_ascii_uppercase();
        if matches!(
            status_upper.as_str(),
            "DONE" | "COMPLETED" | "SUCCESS" | "ERROR" | "FAILED" | "CANCELLED"
        ) {
            return Ok(status_value.to_string());
        }
        sleep(poll_interval);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api_key = args
        .api_key
        .or_else(|| env::var("IBM_API_KEY").ok())
        .context("IBM API key missing: pass --api-key or IBM_API_KEY")?;
    let service_crn = args
        .service_crn
        .or_else(|| env::var("IBM_INSTANCE_CRN").ok())
        .context("IBM instance CRN missing: pass --service-crn or IBM_INSTANCE_CRN")?;
    let matrix: PayloadMatrix = serde_json::from_slice(
        &fs::read(&args.payload_matrix)
            .with_context(|| format!("failed to read {:?}", args.payload_matrix))?,
    )
    .context("failed to decode payload matrix")?;

    if matrix.schema != "scpn_ibm_runtime_sampler_payload_matrix_v1" {
        bail!("unexpected payload matrix schema: {}", matrix.schema);
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout_s))
        .build()
        .context("failed to construct HTTP client")?;

    let token = fetch_iam_token(&api_key, &client)?;
    let headers = build_headers(&token, &service_crn)?;

    let mut rows: Vec<RunRow> = Vec::with_capacity(matrix.rows.len());
    for row in matrix.rows {
        let submit_started = Instant::now();
        let job_id = submit_job(&client, &args.runtime_url, &headers, &row.payload)?;
        let submit_overhead = submit_started.elapsed().as_secs_f64();
        let final_status = poll_status(
            &client,
            &args.runtime_url,
            &headers,
            &job_id,
            Duration::from_secs(args.timeout_s),
            Duration::from_secs_f64(args.poll_interval_s.max(0.1)),
        )?;
        let submit_to_done = submit_started.elapsed().as_secs_f64();
        rows.push(RunRow {
            lane: row.lane,
            scenario: row.scenario,
            job_id,
            submit_overhead_seconds: submit_overhead,
            submit_to_done_seconds: submit_to_done,
            final_status,
        });
    }

    let report = RunReport {
        schema: "scpn_ibm_runtime_rust_latency_run_v1".to_string(),
        backend: matrix.backend,
        runner: "scpn_quantum_engine::ibm_runtime_latency_runner".to_string(),
        rows,
    };
    let output_bytes = serde_json::to_vec_pretty(&report).context("failed to encode run report")?;
    fs::write(&args.output, output_bytes)
        .with_context(|| format!("failed to write {:?}", args.output))?;
    Ok(())
}
