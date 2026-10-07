#!/usr/bin/env python3
"""Starts and stops this robot's modules (roslaunch child processes) on request.

Runs in the robot's namespace (/<robot>/module_manager) and offers
  ~command  (v4w_manager/ModuleCommand)  start | stop | restart a module, or stop "all"
  ~log      (v4w_manager/ModuleLog)      recent output of a module
  ~status   (v4w_manager/ModuleStatus)   state of every module, 2 Hz, latched
Modules come from a YAML file (see config/modules.yaml).
"""
import collections
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
import xmlrpc.client

import rosgraph
import rospy
import yaml
from geometry_msgs.msg import Twist
from v4w_manager.msg import ModuleState, ModuleStatus
from v4w_manager.srv import ModuleCommand, ModuleCommandResponse, ModuleLog, ModuleLogResponse

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STOP_TIMEOUTS = ((signal.SIGINT, 15.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0))
MASTER_LOSS_TIMEOUT = 10.0
MASTER_CHECK_PERIOD = 1.0


class _TimeoutTransport(xmlrpc.client.Transport):
    """XML-RPC with a timeout, so a dead network cannot block the master watchdog for minutes."""

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 3.0
        return connection


class Module:
    def __init__(self, name, cfg, robot, log_dir):
        self.name = name
        self.cmd = shlex.split(cfg["cmd"].replace("${ROBOT}", robot).replace("$ROBOT", robot))
        self.requires = list(cfg.get("requires", []))
        self.health_topic = cfg.get("health_topic")
        self.health_timeout = float(cfg.get("health_timeout", 2.0))
        self.start_grace = float(cfg.get("start_grace", 30.0))
        self.zero_cmd_on_stop = bool(cfg.get("zero_cmd_on_stop", False))
        self.log_path = os.path.join(log_dir, name + ".log")
        self.log = collections.deque(maxlen=500)
        self.proc = None
        self.started = None
        self.stopping = False
        self.expected_exit = False  # set by stop(), so the exit is not reported as a failure
        self.failed = False
        self.message = ""
        self.last_health = None  # rospy time of the last health message
        self.lock = threading.Lock()  # serialises start/stop of this module

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def health_age(self):
        if self.last_health is None:
            return -1.0
        return time.monotonic() - self.last_health

    def state(self):
        if self.stopping:
            return ModuleState.STOPPING
        if not self.alive():
            return ModuleState.FAILED if self.failed else ModuleState.STOPPED
        if not self.health_topic:
            return ModuleState.RUNNING
        age = self.health_age()
        if 0 <= age <= self.health_timeout:
            return ModuleState.HEALTHY
        if time.monotonic() - self.started < self.start_grace:
            return ModuleState.STARTING
        return ModuleState.UNHEALTHY

    def up(self):
        """Up enough for modules that require it."""
        return self.state() in (ModuleState.RUNNING, ModuleState.HEALTHY)


class ModuleManager:
    def __init__(self):
        self.robot = rospy.get_param("~robot_name", "") or socket.gethostname()
        config = rospy.get_param("~config")
        log_dir = os.path.expanduser(rospy.get_param("~log_dir", "~/.ros/module_manager"))
        os.makedirs(log_dir, exist_ok=True)

        with open(config) as f:
            cfg = yaml.safe_load(f)["modules"]
        self.modules = collections.OrderedDict(
            (name, Module(name, mcfg, self.robot, log_dir)) for name, mcfg in cfg.items())
        for m in self.modules.values():
            unknown = [r for r in m.requires if r not in self.modules]
            if unknown:
                raise ValueError("module %s requires unknown module(s) %s" % (m.name, unknown))
            if m.health_topic:
                rospy.Subscriber(m.health_topic, rospy.AnyMsg, self._on_health, m, queue_size=1)

        self.cmd_pub = rospy.Publisher("/%s/cmd_vel" % self.robot, Twist, queue_size=1)
        self.status_pub = rospy.Publisher("~status", ModuleStatus, queue_size=1, latch=True)
        rospy.Service("~command", ModuleCommand, self._on_command)
        rospy.Service("~log", ModuleLog, self._on_log)
        self.master = xmlrpc.client.ServerProxy(rosgraph.get_master_uri(), transport=_TimeoutTransport())
        self.master_id = self._master_identity()
        threading.Thread(target=self._watch_master, daemon=True).start()
        rospy.Timer(rospy.Duration(0.5), self._tick)
        rospy.on_shutdown(self.stop_all)
        rospy.loginfo("module manager for %s: %s", self.robot, ", ".join(self.modules))

    # ---- services ----
    def _on_command(self, req):
        action, name = req.action.strip().lower(), req.module.strip()
        if name == "all":
            if action != "stop":
                return ModuleCommandResponse(False, "only 'stop' works on all modules")
            self.stop_all()
            return ModuleCommandResponse(True, "stopped all modules")
        if name not in self.modules:
            return ModuleCommandResponse(False, "unknown module '%s'" % name)
        if action == "start":
            ok, msg = self.start(self.modules[name])
        elif action == "stop":
            stopped = self.stop(self.modules[name])
            ok, msg = True, ("stopped " + ", ".join(stopped)) if stopped else "%s was not running" % name
        elif action == "restart":
            self.stop(self.modules[name], cascade=False)
            ok, msg = self.start(self.modules[name])
        else:
            ok, msg = False, "unknown action '%s' (start, stop, restart)" % action
        return ModuleCommandResponse(ok, msg)

    def _on_log(self, req):
        if req.module not in self.modules:
            return ModuleLogResponse("unknown module '%s'" % req.module)
        lines = list(self.modules[req.module].log)
        if req.lines:
            lines = lines[-req.lines:]
        return ModuleLogResponse("\n".join(lines))

    # ---- start / stop ----
    def start(self, m):
        with m.lock:
            if m.alive():
                return True, "%s already running" % m.name
            missing = [r for r in m.requires if not self.modules[r].up()]
            if missing:
                return False, "%s needs %s running first" % (m.name, ", ".join(missing))
            m.log.clear()
            m.last_health = None
            m.failed = False
            m.expected_exit = False
            m.message = ""
            env = dict(os.environ, PYTHONUNBUFFERED="1", ROSCONSOLE_STDOUT_LINE_BUFFERED="1")
            # roslaunch gives this node ROS_NAMESPACE=/<robot>; children must not inherit it,
            # their launch files add the robot namespace themselves.
            env.pop("ROS_NAMESPACE", None)
            try:
                logfile = open(m.log_path, "w")
                m.proc = subprocess.Popen(m.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            except OSError as e:
                m.message = "could not start: %s" % e
                m.failed = True
                return False, "%s %s" % (m.name, m.message)
            m.started = time.monotonic()
            threading.Thread(target=self._pump, args=(m, m.proc, logfile), daemon=True).start()
            rospy.loginfo("started %s (pid %d): %s", m.name, m.proc.pid, " ".join(m.cmd))
            return True, "started %s" % m.name

    def stop(self, m, cascade=True):
        """Stop m (and, with cascade, every module that requires it). Returns the names stopped."""
        stopped = []
        if cascade:
            for d in self.modules.values():
                if m.name in d.requires:
                    stopped += self.stop(d)
        with m.lock:
            m.failed = False
            m.message = ""
            if not m.alive():
                return stopped
            m.stopping = True
            m.expected_exit = True
            try:
                if m.zero_cmd_on_stop:
                    self._zero_cmd()
                pgid = os.getpgid(m.proc.pid)
                for sig, timeout in STOP_TIMEOUTS:
                    try:
                        os.killpg(pgid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        m.proc.wait(timeout)
                        break
                    except subprocess.TimeoutExpired:
                        rospy.logwarn("%s still running after %s, escalating", m.name, sig.name)
                if m.zero_cmd_on_stop:
                    self._zero_cmd()
            finally:
                m.stopping = False
            rospy.loginfo("stopped %s", m.name)
        return stopped + [m.name]

    def stop_all(self):
        # Modules nothing else requires go first, so MPPI stops before control.
        for m in reversed(list(self.modules.values())):
            self.stop(m)

    def _zero_cmd(self):
        for _ in range(5):
            self.cmd_pub.publish(Twist())
            time.sleep(0.02)

    # ---- monitoring ----
    def _pump(self, m, proc, logfile):
        with logfile:
            for raw in iter(proc.stdout.readline, b""):
                line = ANSI.sub("", raw.decode(errors="replace").rstrip())
                m.log.append(line)
                logfile.write(line + "\n")
                logfile.flush()
        code = proc.wait()
        if m.proc is proc and not m.expected_exit:
            m.failed = True
            m.message = "exited with code %d" % code
            rospy.logwarn("%s exited on its own (code %d)", m.name, code)

    def _on_health(self, _msg, m):
        if m.alive():
            m.last_health = time.monotonic()

    def _tick(self, _event):
        status = ModuleStatus(robot=self.robot)
        status.header.stamp = rospy.Time.now()
        for m in self.modules.values():
            status.modules.append(ModuleState(
                name=m.name, state=m.state(), requires=m.requires,
                uptime=time.monotonic() - m.started if m.alive() else 0.0,
                health_age=m.health_age() if m.alive() else -1.0,
                message=m.message))
        self.status_pub.publish(status)


    # ---- master watchdog ----
    def _master_identity(self):
        """Something that changes when the master restarts: its run_id (set by roslaunch/roscore), else its pid."""
        caller = rospy.get_name()
        code, _, run_id = self.master.getParam(caller, "/run_id")
        if code == 1:
            return "run_id " + str(run_id)
        return "pid %s" % self.master.getPid(caller)[2]

    def _watch_master(self):
        # Nodes cannot re-register with a restarted master: if the master restarts, or stays
        # gone, stop everything and exit so systemd restarts the manager against the new one.
        lost_since = None
        while not rospy.is_shutdown():
            time.sleep(MASTER_CHECK_PERIOD)
            try:
                identity = self._master_identity()
            except Exception:  # unreachable, refused, timeout
                lost_since = lost_since or time.monotonic()
                if time.monotonic() - lost_since > MASTER_LOSS_TIMEOUT:
                    reason = "master unreachable for %.0f s" % MASTER_LOSS_TIMEOUT
                    break
                continue
            lost_since = None
            if identity != self.master_id:
                reason = "master restarted (%s -> %s)" % (self.master_id, identity)
                break
        else:
            return
        rospy.logerr("%s: stopping all modules and exiting", reason)
        self._exit_on_master_loss()

    def _exit_on_master_loss(self):
        self.stop_all()
        os._exit(1)


if __name__ == "__main__":
    rospy.init_node("module_manager")
    ModuleManager()
    rospy.spin()
