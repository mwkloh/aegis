# HEARTBEAT.md - Periodic Checklist

Read by the reflection plane during a scheduled sweep. Everything here is optional: delete what you don't want, add what you do.

## System integrity

- Is Ollama reachable, and are the configured models installed? (`make doctor`)
- Did any tool fail repeatedly in the last day? Check the event log under `sessions/`.

## Memory health

- Is `memory/MEMORY.md` still short and accurate? Remove stale lines.
- Are there duplicated facts?

## Security

- Any unexpected `.env` or config changes?
- Any confirmation prompts that were approved without being read?

## Proactive check

Is there something the operator would want to hear about today? If not, stay quiet.
