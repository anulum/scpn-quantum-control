// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — strict verdicts on the declared Studio resource budgets

import budgetText from "../../../../data/studio/resource_budgets_20261123.json?raw";

/** Schema of the committed budget declaration. */
export const budgetsSchema = "scpn_qc_studio_resource_budgets_v1";
/** How a measurement meets its target; `record` has no verdict. */
export type Comparator = "<" | "<=" | ">=" | "record";
/** Verdict of one budget. */
export type BudgetStatus = "pass" | "fail" | "unmeasured" | "recorded";

/** One declared budget. */
export interface Budget {
  /** Stable identifier. */ readonly id: string;
  /** What is measured. */ readonly metric: string;
  /** How the measurement meets the target. */ readonly comparator: Comparator;
  /** The target, or null for a record-only budget. */ readonly target: number | null;
  /** Unit of target and measurement. */ readonly unit: string;
}
/** The verdict on one budget. */
export interface BudgetVerdict {
  /** The budget. */ readonly budget: Budget;
  /** The measurement, or null when unmeasured. */ readonly measured: number | null;
  /** The verdict. */ readonly status: BudgetStatus;
}

const comparators: readonly Comparator[] = ["<", "<=", ">=", "record"];

function budgetOf(value: unknown, path: string): Budget {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`${path} must be an object`);
  const item = value as Readonly<Record<string, unknown>>;
  const comparator = comparators.find((entry) => entry === item["comparator"]);
  if (comparator === undefined)
    throw new Error(`${path}.comparator is not one of ${comparators.join(", ")}`);
  const target = item["target"];
  if ((comparator === "record") !== (target === null))
    throw new Error(`${path}: a record-only budget has no target, and every other has one`);
  if (target !== null && (typeof target !== "number" || !Number.isFinite(target)))
    throw new Error(`${path}.target must be a finite number`);
  const texts = ["id", "metric", "unit"].map((key) => item[key]);
  if (!texts.every((text) => typeof text === "string" && text !== ""))
    throw new Error(`${path} needs a non-empty id, metric and unit`);
  const [id, metric, unit] = texts as string[];
  return { id: id as string, metric: metric as string, comparator, target, unit: unit as string };
}

/**
 * Parse a budget declaration.
 *
 * @param source - The declaration's JSON text.
 * @returns The budgets in declared order.
 * @throws Error when the declaration is malformed or repeats an id.
 */
export function parseBudgets(source: string): readonly Budget[] {
  const document = JSON.parse(source) as unknown;
  if (
    typeof document !== "object" ||
    document === null ||
    (document as Record<string, unknown>)["schema"] !== budgetsSchema
  )
    throw new Error(`the declaration must be a ${budgetsSchema} object`);
  const items = (document as Record<string, unknown>)["budgets"];
  if (!Array.isArray(items) || items.length === 0)
    throw new Error("$.budgets must be a non-empty array");
  const budgets = items.map((item, index) => budgetOf(item, `$.budgets[${index}]`));
  if (new Set(budgets.map((budget) => budget.id)).size !== budgets.length)
    throw new Error("$.budgets repeats an id");
  return budgets;
}

/** The budgets committed with this build. */
export const committedBudgets: readonly Budget[] = parseBudgets(budgetText);

/**
 * Judge one measurement strictly; no tolerance, retry or relaxed threshold exists.
 *
 * @param budget - The budget.
 * @param measured - A finite measurement, or null when it was not measured.
 * @returns `unmeasured` without a measurement, `recorded` for a record-only budget, otherwise
 *   `pass` exactly when the comparator holds.
 * @throws Error when the measurement is not finite.
 */
export function judgeBudget(budget: Budget, measured: number | null): BudgetVerdict {
  if (measured === null) return { budget, measured, status: "unmeasured" };
  if (!Number.isFinite(measured)) throw new Error(`${budget.id} must be a finite number`);
  if (budget.comparator === "record" || budget.target === null)
    return { budget, measured, status: "recorded" };
  const holds =
    budget.comparator === "<"
      ? measured < budget.target
      : budget.comparator === "<="
        ? measured <= budget.target
        : measured >= budget.target;
  return { budget, measured, status: holds ? "pass" : "fail" };
}

/**
 * Look up a committed budget by id.
 *
 * @param id - The budget id.
 * @returns The budget.
 * @throws Error when no committed budget has that id.
 */
export function committedBudget(id: string): Budget {
  const budget = committedBudgets.find((item) => item.id === id);
  if (budget === undefined) throw new Error(`no committed budget ${id}`);
  return budget;
}
