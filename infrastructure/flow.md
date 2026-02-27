# Infrastructure Flow

The `compute/infrastructure/` directory provides the **System Utilities** that enable safe, cross-platform execution of external processes.

## Key Files & Responsibilities

### 1. Subprocess Utility (`subprocess.h` / `.cpp`)
**Role**: **The Sandbox**. A POSIX-based wrapper for spawning, managing, and monitoring external EDA tools.
-   **Resource Monitoring**: Implements timeouts to prevent zombie processes.
-   **Output Streaming**: Uses pipes and `select()` (or `epoll`) to stream stdout/stderr in real-time.
-   **Isolation**: Provides hooks for setting working directories and environment variables to ensure tool isolation.

## Execution Pipeline

1.  **Request**: `WorkerService` or a Driver requests a command execution.
2.  **Creation**: `Subprocess` forks the current process and calls `execvp`.
3.  **Monitoring**: The parent process sits in a `select()` loop, reading from pipes and checking the `timeout` clock.
4.  **Feedback**: Lines read from the pipes are dispatched via callbacks to the requester.
5.  **Completion**: The process exit code is captured and returned.

## Development Rules

1.  **No Blocking**: I/O must be non-blocking or handled in a way that allows for timeout checks.
2.  **Signal Safety**: Ensure child processes are correctly reaped and do not leave orphans on timeout.
3.  **Cross-Platform**: While currently POSIX-centric, the interface should remain generic enough for a future Windows implementation.

## 🤖 SME Validation Checklist
*(Consult this list before modifying `compute/infrastructure/`)*

- [ ] **Timeout Handling**: Does the implementation correctly kill the child process group on timeout?
- [ ] **Pipe Buffering**: Is the buffer size sufficient to handle high-volume tool output without stalling?
- [ ] **Resource Cleanup**: Are file descriptors (pipes) explicitly closed in both parent and child branches?
