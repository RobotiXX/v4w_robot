uint8 STOPPED=0
uint8 STARTING=1    # process up, health topic not seen yet (within the start grace period)
uint8 RUNNING=2     # process up, module has no health topic
uint8 HEALTHY=3     # process up, health topic publishing
uint8 UNHEALTHY=4   # process up, health topic silent for longer than health_timeout
uint8 STOPPING=5
uint8 FAILED=6      # process exited without being asked to

string name
uint8 state
string[] requires
float32 uptime      # s since start, 0 when not running
float32 health_age  # s since the last health topic message, -1 if none
string message      # exit code, why a start was refused, ...
