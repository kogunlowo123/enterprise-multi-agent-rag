# ADR 0001: In-repo workflow engine instead of a graph framework

- Status: Accepted
- Date: 2026-09-19

## Context

The pipeline is a small directed graph with one loop (validate, escalate, retrieve). Graph frameworks
such as LangGraph provide this, plus checkpointing and streaming, at the cost of a large dependency
tree and version churn.

## Decision

Implement a ~75 line `WorkflowGraph` (`graph.py`) with static edges, conditional edges, structural
validation and a step budget. Agents are plain classes with a `run(state)` method, so they are
independent of the engine.

## Consequences

- Zero framework dependencies; tests run in milliseconds and the control flow is readable in one file.
- No built-in checkpointing, streaming or visual tooling.
- Because agents do not import the engine, they should be mountable on LangGraph nodes without changes. An adapter
  is on the roadmap for teams that need those features.

## Alternatives considered

- **LangGraph now.** Rejected for v0.1: the workflow has eight nodes and no durable-execution
  requirement, so the dependency outweighs the benefit.
