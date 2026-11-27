"""Transcription of the m6 friction budget from bam/mjlab.py:471-536 (@62bd8ce), line by line.

Used as the R2-G01 friction oracle because the real module imports mujoco_warp and the
mjlab package (not installed; GPU-only). Every branch below mirrors the cited lines;
the CPU alternative `Model.compute_frictions` (bam/model.py:114-160) lacks the m6
quadratic term, so both oracles are reported.
"""
import numpy as np


def gpu_form_friction_budget(bam, motor_torque, external_torque, dtheta):
    """bam/mjlab.py:471-536 with friction_scale = 1."""
    stribeck_coeff = np.exp(-(np.abs(dtheta / bam.dtheta_stribeck.value) ** bam.alpha.value))  # :657
    frictionloss = np.full_like(np.asarray(motor_torque, dtype=float), bam.friction_base.value)  # :445
    if bam.stribeck:                                                                            # :444
        frictionloss = frictionloss + stribeck_coeff * bam.friction_stribeck.value              # :446
    if bam.load_dependent:
        if bam.directional:                                                                     # :452
            gearbox_torque = np.abs(external_torque * bam.load_friction_external.value          # :454-457
                                    - motor_torque * bam.load_friction_motor.value)
            frictionloss = frictionloss + gearbox_torque                                        # :458
            if bam.stribeck:
                gearbox_torque_stribeck = np.abs(external_torque * bam.load_friction_external_stribeck.value
                                                 - motor_torque * bam.load_friction_motor_stribeck.value)  # :461-464
                frictionloss = frictionloss + stribeck_coeff * gearbox_torque_stribeck           # :465-467
                if bam.quadratic:                                                                # :469
                    abs_ext = np.abs(external_torque)                                            # :471
                    abs_mot = np.abs(motor_torque)                                               # :472
                    drive_mask = (abs_mot > abs_ext).astype(float)                               # :473
                    backdrive_mask = 1.0 - drive_mask                                            # :474
                    quad_term = (drive_mask * bam.load_friction_external_quad.value * abs_ext ** 2
                                 + backdrive_mask * bam.load_friction_motor_quad.value * abs_mot ** 2)  # :475-481
                    frictionloss = frictionloss + stribeck_coeff * quad_term                     # :482
    return frictionloss