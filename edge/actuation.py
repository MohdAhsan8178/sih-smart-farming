"""
edge/actuation.py
-----------------
Fail-Safe Actuation and Flow-Monitored Irrigation Control Loop.

SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system
produces irrigation prescriptions for manual farmer execution. This module
is retained as a validated reference implementation for future closed-loop
deployment and is NOT wired into any runtime path.

Implements:
  1. Fail-safe hardware control (Normally Closed valves, MOSFET pull-down states).
  2. Software safety loop with try...finally GPIO de-assertion.
  3. Continuous flow feedback & fault detection:
     - FAULT_NO_FLOW: Commanded OPEN but zero flow detected (dry-run / pump cavitation / pipe blockage).
     - FAULT_LEAK_DETECTED: Commanded CLOSED but flow pulses detected (valve stuck open / seal rupture).
     - FAULT_WATCHDOG_TIMEOUT: Continuous irrigation exceeds maximum allowable session duration.
  4. Context manager ensuring fail-safe shutdown upon any unhandled exception or exit.

Hardware Safety Architecture:
  - Normally Closed (NC) Solenoid Valves: In the event of catastrophic power loss,
    mechanical return springs seat the valve diaphragm, preventing uncontrolled field flooding.
  - MOSFET Gate Pull-Downs (10k ohm to GND): When microcontroller / Jetson GPIOs float during
    bootup, kernel panic, or software crash, gates remain LOW to keep switches non-conductive.

Provisional Safeguard Constants:
  - PROVISIONAL_MAX_IRRIGATION_DURATION_S: Maximum continuous open time before forced watchdog cutoff.
  - PROVISIONAL_FLOW_DISAGREEMENT_TIMEOUT_S: Seconds of zero flow while open before tripping dry-run fault.
  - PROVISIONAL_LEAK_PULSE_THRESHOLD: Pulses observed while closed that flag a hydraulic leak.
"""

from typing import Dict, Any, Optional, List
import time


# ==============================================================================
# Provisional Safety Constants
# ==============================================================================
PROVISIONAL_MAX_IRRIGATION_DURATION_S: float = 3600.0     # 1 hour maximum continuous runtime
PROVISIONAL_FLOW_DISAGREEMENT_TIMEOUT_S: float = 15.0     # 15 seconds zero flow -> trip dry-run fault
PROVISIONAL_LEAK_PULSE_THRESHOLD: int = 5                 # >= 5 pulses while valve closed -> trip leak fault
PROVISIONAL_MIN_EXPECTED_RATE_LPM: float = 0.5            # Flow rate floor when open


# ==============================================================================
# Mock GPIO Interface (Decoupled from Physical Jetson.GPIO)
# ==============================================================================

class MockGPIO:
    """
    Mock GPIO interface for synthetic execution and unit testing without
    physical Jetson.GPIO / RPi.GPIO hardware access.
    """
    LOW = 0
    HIGH = 1

    def __init__(self):
        self.pin_states: Dict[int, int] = {}
        self.cleaned_up: bool = False

    def setup(self, pin: int, mode: Any) -> None:
        self.pin_states[pin] = self.LOW

    def output(self, pin: int, state: int) -> None:
        self.pin_states[pin] = 1 if state else 0

    def input(self, pin: int) -> int:
        return self.pin_states.get(pin, self.LOW)

    def get_pin(self, pin: int) -> int:
        """Alias for input(pin) returning 0 or 1."""
        return self.input(pin)

    def cleanup(self) -> None:
        for pin in self.pin_states:
            self.pin_states[pin] = self.LOW
        self.cleaned_up = True


# ==============================================================================
# Fail-Safe Actuation Controller
# ==============================================================================

class ActuationController:
    """
    Coordinates fail-safe valve and pump actuation with continuous flow monitoring.
    """
    STATE_IDLE = "IDLE"
    STATE_IRRIGATING = "IRRIGATING"
    STATE_FAULT_NO_FLOW = "FAULT_NO_FLOW"
    STATE_FAULT_LEAK = "FAULT_LEAK_DETECTED"
    STATE_FAULT_WATCHDOG = "FAULT_WATCHDOG_TIMEOUT"
    STATE_EMERGENCY_STOP = "EMERGENCY_STOP"

    def __init__(self,
                 pump_pin: int,
                 valve_pins: Dict[str, int],
                 gpio_backend: Optional[Any] = None,
                 max_duration_s: float = PROVISIONAL_MAX_IRRIGATION_DURATION_S,
                 no_flow_timeout_s: float = PROVISIONAL_FLOW_DISAGREEMENT_TIMEOUT_S,
                 leak_pulse_thresh: int = PROVISIONAL_LEAK_PULSE_THRESHOLD):
        self.pump_pin = int(pump_pin)
        self.valve_pins = {k: int(v) for k, v in valve_pins.items()}
        self.gpio = gpio_backend if gpio_backend is not None else MockGPIO()

        self.max_duration_s = float(max_duration_s)
        self.no_flow_timeout_s = float(no_flow_timeout_s)
        self.leak_pulse_thresh = int(leak_pulse_thresh)

        self.state: str = self.STATE_IDLE
        self.active_zone: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.last_flow_time: Optional[float] = None
        self.closed_leak_pulses: int = 0
        self.fault_reason: Optional[str] = None

        # Initialize hardware pins to safe LOW state
        self.deassert_all()

    def deassert_all(self) -> None:
        """
        Immediately drive all pump and valve pins LOW (Normally Closed / Safe).
        Can be called from signal handlers or exception blocks.
        """
        self.gpio.output(self.pump_pin, self.gpio.LOW)
        for vpin in self.valve_pins.values():
            self.gpio.output(vpin, self.gpio.LOW)

    def start_irrigation(self, zone: str, start_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Open zone valve and start main pump.
        """
        if self.state in (self.STATE_FAULT_NO_FLOW, self.STATE_FAULT_LEAK,
                          self.STATE_FAULT_WATCHDOG, self.STATE_EMERGENCY_STOP):
            raise RuntimeError(f"Cannot start irrigation while controller is in fault state '{self.state}': {self.fault_reason}")

        if zone not in self.valve_pins:
            raise ValueError(f"Unknown zone '{zone}'. Configured zones: {list(self.valve_pins.keys())}")

        now = float(start_time if start_time is not None else time.time())

        # First assert zone valve (prevent pump running against closed deadhead)
        self.gpio.output(self.valve_pins[zone], self.gpio.HIGH)
        # Then assert pump
        self.gpio.output(self.pump_pin, self.gpio.HIGH)

        self.state = self.STATE_IRRIGATING
        self.active_zone = zone
        self.session_start_time = now
        self.last_flow_time = now
        self.closed_leak_pulses = 0
        self.fault_reason = None

        return {
            "status": "started",
            "state": self.state,
            "zone": zone,
            "start_time": now
        }

    def stop_irrigation(self, reason: str = "normal_completion") -> Dict[str, Any]:
        """
        Safely shutdown pump and all zone valves.
        """
        # First shut off pump to eliminate line pressure
        self.gpio.output(self.pump_pin, self.gpio.LOW)
        # Then close valves
        for vpin in self.valve_pins.values():
            self.gpio.output(vpin, self.gpio.LOW)

        previous_zone = self.active_zone
        self.active_zone = None
        self.session_start_time = None
        self.last_flow_time = None
        self.state = self.STATE_IDLE

        return {
            "status": "stopped",
            "state": self.state,
            "zone": previous_zone,
            "reason": reason
        }

    def emergency_stop(self, reason: str = "user_e_stop") -> Dict[str, Any]:
        """
        Immediate emergency shutdown and fault latching.
        """
        self.deassert_all()
        self.state = self.STATE_EMERGENCY_STOP
        self.fault_reason = str(reason)
        return {
            "status": "emergency_stopped",
            "state": self.state,
            "reason": self.fault_reason
        }

    def reset_fault(self) -> None:
        """Clear latched fault after inspection."""
        self.deassert_all()
        self.state = self.STATE_IDLE
        self.fault_reason = None
        self.closed_leak_pulses = 0

    def update_safety_loop(self,
                           current_time: float,
                           flow_pulses_delta: int = 0,
                           flow_rate_lpm: float = 0.0) -> Dict[str, Any]:
        """
        Periodic safety supervisor step.
        Checks:
          1. Watchdog timeout during irrigation.
          2. No-flow dry-run fault when commanded open.
          3. Leak detection when commanded closed.
        """
        now = float(current_time)

        # ----------------------------------------------------------------------
        # Case A: Controller is actively IRRIGATING
        # ----------------------------------------------------------------------
        if self.state == self.STATE_IRRIGATING:
            # 1. Watchdog check
            elapsed = now - (self.session_start_time or now)
            if elapsed > self.max_duration_s:
                self.deassert_all()
                self.state = self.STATE_FAULT_WATCHDOG
                self.fault_reason = f"Max continuous runtime ({self.max_duration_s}s) exceeded. Elapsed: {elapsed:.1f}s."
                return {"state": self.state, "fault": self.fault_reason, "action": "tripped_watchdog"}

            # 2. Flow disagreement check (pump ON, but no pulses)
            if flow_pulses_delta > 0 or flow_rate_lpm >= PROVISIONAL_MIN_EXPECTED_RATE_LPM:
                self.last_flow_time = now
            else:
                stalled_time = now - (self.last_flow_time or now)
                if stalled_time > self.no_flow_timeout_s:
                    self.deassert_all()
                    self.state = self.STATE_FAULT_NO_FLOW
                    self.fault_reason = (f"No water flow detected for {stalled_time:.1f}s "
                                         f"(limit: {self.no_flow_timeout_s}s) while pump active.")
                    return {"state": self.state, "fault": self.fault_reason, "action": "tripped_no_flow"}

            return {
                "state": self.state,
                "elapsed_s": elapsed,
                "zone": self.active_zone,
                "flow_ok": True
            }

        # ----------------------------------------------------------------------
        # Case B: Controller is IDLE (Commanded CLOSED)
        # ----------------------------------------------------------------------
        elif self.state == self.STATE_IDLE:
            # Leak check: pulses observed while system is supposed to be off
            if flow_pulses_delta > 0:
                self.closed_leak_pulses += int(flow_pulses_delta)
                if self.closed_leak_pulses >= self.leak_pulse_thresh:
                    self.deassert_all()
                    self.state = self.STATE_FAULT_LEAK
                    self.fault_reason = (f"Hydraulic leak detected: {self.closed_leak_pulses} pulses "
                                         f"observed while valves commanded closed.")
                    return {"state": self.state, "fault": self.fault_reason, "action": "tripped_leak"}

            return {
                "state": self.state,
                "closed_leak_pulses": self.closed_leak_pulses,
                "leak_fault": False
            }

        # ----------------------------------------------------------------------
        # Case C: Fault / E-Stop latched
        # ----------------------------------------------------------------------
        else:
            self.deassert_all()  # Continuously enforce deassertion
            return {
                "state": self.state,
                "fault": self.fault_reason,
                "latched": True
            }

    def session(self, zone: str, start_time: Optional[float] = None):
        """
        Context manager helper providing guaranteed try...finally de-assertion.

        Usage:
          with controller.session("zone_1"):
              # perform irrigation
        """
        return _IrrigationSessionContext(self, zone, start_time)


class _IrrigationSessionContext:
    """Context manager wrapper for ActuationController."""
    def __init__(self, controller: ActuationController, zone: str, start_time: Optional[float]):
        self.controller = controller
        self.zone = zone
        self.start_time = start_time

    def __enter__(self):
        self.controller.start_irrigation(self.zone, start_time=self.start_time)
        return self.controller

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Guaranteed fail-safe de-assertion regardless of normal exit or exception
        self.controller.stop_irrigation(reason="context_exit" if exc_type is None else "context_exception")
        return False  # Do not suppress exceptions
