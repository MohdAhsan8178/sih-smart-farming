"""
edge/flow.py
------------
YF-S201 Hall-Effect Turbine Water Flow Sensor Interface.

SCOPE NOTE (7 Sep 2026): Actuation hardware is out of scope. The system
produces irrigation prescriptions for manual farmer execution. This module
is retained as a validated reference implementation for future closed-loop
deployment and is NOT wired into any runtime path.

Implements:
  1. Pure pulse-to-liter and frequency-to-LPM conversion logic.
  2. FlowMonitor accumulator tracking delivered irrigation volume,
     instantaneous flow rate, and target volume completion.
  3. Delivery volume verification and tolerance checking.

Hardware & Physical Specifications:
  - Sensor: YF-S201 1/2" Hall-effect water flow turbine.
  - Nominal characteristic:
      Pulse frequency (Hz) = 7.5 * Q (L/min)
      Therefore:
      Pulses per liter = 7.5 * 60 s = 450 pulses/L (~2.222 mL per pulse).
  - Working flow range: 1 to 30 L/min.
  - Working pressure: <= 1.75 MPa.

Caution / Guard:
  - PROVISIONAL_YF_S201_PULSES_PER_LITER = 450.0 is the manufacturer's nominal rating.
    Due to injection-molded turbine tolerances (+/- 10%), line pressure, pipe diameter
    mismatches, and water viscosity/temperature, physical volumetric calibration using a
    graduated vessel is mandatory before precision irrigation delivery.
"""

from typing import Dict, Any, Optional


# ==============================================================================
# Provisional Sensor Constants
# ==============================================================================
PROVISIONAL_YF_S201_PULSES_PER_LITER: float = 450.0   # Nominal pulses per liter
PROVISIONAL_YF_S201_HZ_PER_LPM: float = 7.5           # Nominal Hz per (L/min)


# ==============================================================================
# Pure Conversion Functions
# ==============================================================================

def pulses_to_liters(pulse_count: int,
                     pulses_per_liter: float = PROVISIONAL_YF_S201_PULSES_PER_LITER) -> float:
    """
    Convert accumulated flow sensor pulses to volume in liters.

    Parameters:
      pulse_count: Total integer pulses counted by interrupt or GPIO counter.
      pulses_per_liter: K-factor (pulses/L).

    Returns:
      Volume in liters (float, >= 0.0).
    """
    if pulse_count < 0:
        raise ValueError(f"pulse_count cannot be negative, got {pulse_count}.")
    if pulses_per_liter <= 0.0:
        raise ValueError(f"pulses_per_liter must be strictly positive, got {pulses_per_liter}.")
    return float(round(float(pulse_count) / float(pulses_per_liter), 4))


def liters_to_pulses(volume_liters: float,
                     pulses_per_liter: float = PROVISIONAL_YF_S201_PULSES_PER_LITER) -> int:
    """
    Convert desired volume in liters to expected target pulses.
    """
    if volume_liters < 0.0:
        raise ValueError(f"volume_liters cannot be negative, got {volume_liters}.")
    if pulses_per_liter <= 0.0:
        raise ValueError(f"pulses_per_liter must be strictly positive, got {pulses_per_liter}.")
    return int(round(float(volume_liters) * float(pulses_per_liter)))


def frequency_to_flow_rate_lpm(frequency_hz: float,
                               hz_per_lpm: float = PROVISIONAL_YF_S201_HZ_PER_LPM) -> float:
    """
    Convert instantaneous pulse frequency (Hz) to flow rate in liters per minute (L/min).
    """
    if frequency_hz < 0.0:
        raise ValueError(f"frequency_hz cannot be negative, got {frequency_hz}.")
    if hz_per_lpm <= 0.0:
        raise ValueError(f"hz_per_lpm must be strictly positive, got {hz_per_lpm}.")
    return float(round(float(frequency_hz) / float(hz_per_lpm), 3))


# ==============================================================================
# Flow Monitor & Accumulator Class
# ==============================================================================

class FlowMonitor:
    """
    Accumulates flow sensor pulses, computes delivered volume and instantaneous rate,
    and validates target delivery goals.
    """
    def __init__(self,
                 pulses_per_liter: float = PROVISIONAL_YF_S201_PULSES_PER_LITER,
                 hz_per_lpm: float = PROVISIONAL_YF_S201_HZ_PER_LPM):
        self.pulses_per_liter = pulses_per_liter
        self.hz_per_lpm = hz_per_lpm

        self.total_pulses: int = 0
        self.delivered_liters: float = 0.0
        self.current_rate_lpm: float = 0.0
        self.last_update_ts: Optional[float] = None

    def reset(self) -> None:
        """Reset volume and pulse accumulator for a new irrigation session."""
        self.total_pulses = 0
        self.delivered_liters = 0.0
        self.current_rate_lpm = 0.0
        self.last_update_ts = None

    def record_pulse_delta(self,
                           delta_pulses: int,
                           elapsed_seconds: float,
                           timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Record a batch of pulses over a measured time interval.

        Parameters:
          delta_pulses: Number of pulses received during the interval.
          elapsed_seconds: Duration of the interval in seconds.
          timestamp: Optional reference timestamp.

        Returns:
          Dict containing updated session status.
        """
        if delta_pulses < 0:
            raise ValueError(f"delta_pulses cannot be negative, got {delta_pulses}.")
        if elapsed_seconds <= 0.0:
            raise ValueError(f"elapsed_seconds must be positive, got {elapsed_seconds}.")

        self.total_pulses += int(delta_pulses)
        self.delivered_liters = pulses_to_liters(self.total_pulses, self.pulses_per_liter)

        freq_hz = float(delta_pulses) / float(elapsed_seconds)
        self.current_rate_lpm = frequency_to_flow_rate_lpm(freq_hz, self.hz_per_lpm)
        if timestamp is not None:
            self.last_update_ts = float(timestamp)

        return {
            "total_pulses": self.total_pulses,
            "delivered_liters": self.delivered_liters,
            "current_rate_lpm": self.current_rate_lpm,
            "interval_hz": float(round(freq_hz, 2)),
            "last_update_ts": self.last_update_ts
        }

    def is_target_reached(self, target_liters: float) -> bool:
        """Check if accumulated delivered volume has met or exceeded target."""
        if target_liters <= 0.0:
            return True
        return bool(self.delivered_liters >= float(target_liters))

    def verify_delivery(self,
                        target_liters: float,
                        tolerance_pct: float = 0.05) -> Dict[str, Any]:
        """
        Verify if delivered irrigation volume matches target within tolerance bounds.

        Parameters:
          target_liters: Desired irrigation volume [L].
          tolerance_pct: Allowable error fraction (default 0.05 = +/- 5%).

        Returns:
          Dict with verification details and audit status.
        """
        if target_liters <= 0.0:
            raise ValueError(f"target_liters must be strictly positive, got {target_liters}.")
        if tolerance_pct < 0.0 or tolerance_pct > 1.0:
            raise ValueError(f"tolerance_pct must be in [0, 1], got {tolerance_pct}.")

        delivered = self.delivered_liters
        discrepancy = delivered - target_liters
        error_pct = (discrepancy / target_liters) * 100.0
        within_bounds = abs(discrepancy) <= (target_liters * tolerance_pct)

        return {
            "target_liters": float(round(target_liters, 3)),
            "delivered_liters": float(round(delivered, 3)),
            "discrepancy_liters": float(round(discrepancy, 3)),
            "error_pct": float(round(error_pct, 2)),
            "tolerance_pct": float(round(tolerance_pct * 100.0, 2)),
            "within_bounds": bool(within_bounds),
            "pulses_counted": self.total_pulses,
            "k_factor": self.pulses_per_liter
        }
