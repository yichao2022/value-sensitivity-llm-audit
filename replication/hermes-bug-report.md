# Gateway dies on macOS restart but launchd doesn't relaunch (zombie "running" state)

## Description

On macOS with launchd, the Hermes gateway sometimes dies during a self-restart cycle but launchd fails to detect the process exit, leaving the service stuck in a "running" state with a dead process. The gateway.log stops updating while `launchctl print` shows `state = running` with a stale PID.

## Environment

- **Hermes Agent**: v0.16.0 (2026.6.5) · upstream c6b0eb4d
- **Python**: 3.11.15
- **OS**: macOS (launchd-managed service)
- **Platform**: Telegram

## Steps to Reproduce

1. Have the gateway running under launchd with `KeepAlive: true`
2. Trigger a gateway restart (e.g., via `/new` command or `hermes gateway restart`)
3. The gateway logs "Stopping gateway for restart..." and exits cleanly
4. launchd does **not** relaunch the gateway
5. `launchctl print` still shows `state = running` with the old (dead) PID
6. `launchctl list | grep hermes` returns nothing
7. Gateway is completely dead — no Telegram polling, no responses

## Observed Behavior

From `gateway.log`:
```
19:40:09 Stopping gateway for restart...
19:40:10 Gateway stopped (total teardown 0.66s)
19:40:10 Cron ticker stopped
```

From `launchctl print`:
```
state = running
pid = 55543    ← dead process, no longer exists
last exit code = 0
```

The service is in a zombie state: launchd thinks it's running, but the process is dead and no new gateway starts.

## Workaround

Requires manual recovery:
```bash
launchctl remove ai.hermes.gateway
# Wait for stale state to clear
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

## Suspected Root Cause

In `gateway/run.py:6791-6798`, on macOS the restart uses `GATEWAY_SERVICE_RESTART_EXIT_CODE = 75`:

```python
self._exit_code = (
    GATEWAY_SERVICE_RESTART_EXIT_CODE
    if sys.platform == "darwin" or not os.environ.get("INVOCATION_ID")
    else 0
)
```

The comment says "launchd's KeepAlive.SuccessfulExit=false needs a non-zero exit to relaunch", but the plist doesn't set `SuccessfulExit` — it only has `<key>KeepAlive</key><true/>`. There may be a race condition where launchd sees the exit code 75, interprets it as a permanent failure, and applies backoff/retry limits rather than immediately relaunching. Or the SIGTERM handler and the self-restart codepath race, causing the process to exit in a way launchd can't track.

## Suggested Fix

1. Set `<key>SuccessfulExit</key><false/>` in the generated launchd plist so launchd treats **any** exit (including code 75) as a crash and immediately relaunches
2. Or: on macOS, exit with code 0 on planned restarts (matching the systemd path) since `KeepAlive: true` unconditionally restarts on clean exits
3. Add a watchdog/heartbeat mechanism that detects the zombie state and self-heals
