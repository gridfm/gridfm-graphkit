# =========================
# === BUS FEATURE INDICES ==
# =========================
PD_H = 0  # Active power demand (P_d)
QD_H = 1  # Reactive power demand (Q_d)
QG_H = 2  # Reactive power generation (Q_g)
VM_H = 3  # Voltage magnitude (p.u.)
VA_H = 4  # Voltage angle (degrees)
PQ_H = 5  # PQ bus indicator (1 if PQ)
PV_H = 6  # PV bus indicator (1 if PV)
REF_H = 7  # Reference (slack) bus indicator (1 if REF)
MIN_VM_H = 8  # Minimum voltage magnitude limit (p.u.)
MAX_VM_H = 9  # Maximum voltage magnitude limit (p.u.)
MIN_QG_H = 10  # Minimum reactive power limit (Mvar)
MAX_QG_H = 11  # Maximum reactive power limit (Mvar)
GS = 12  # Shunt conductance (p.u.)
BS = 13  # Shunt susceptance (p.u.)
VN_KV = 14  # Nominal voltage

# =========================
# === OUTPUT FEATURE INDICES ==
# =========================
VM_OUT = 0
VA_OUT = 1
PG_OUT = 2
QG_OUT = 3
PD_OUT = 4  # for random masking
QD_OUT = 5  # for random masking
PG_OUT_GEN = 0


# ================================
# === GENERATOR FEATURE INDICES ==
# ================================
PG_H = 0  # Active power generation (P_g)
MIN_PG = 1  # Minimum active power limit (MW)
MAX_PG = 2  # Maximum active power limit (MW)
C0_H = 3  # Cost coefficient c0 (€)
C1_H = 4  # Cost coefficient c1 (€ / MW)
C2_H = 5  # Cost coefficient c2 (€ / MW²)
G_ON = 6  # Generator on/off

# ============================
# === EDGE FEATURE INDICES ===
# ============================
P_E = 0  # Active power flow
Q_E = 1  # Reactive power flow
YFF_TT_R = 2  # Yff real
YFF_TT_I = 3  # Yff imag
YFT_TF_R = 4  # Yft real
YFT_TF_I = 5  # Yft imag
TAP = 6  # Tap ratio
ANG_MIN = 7  # Angle min (deg)
ANG_MAX = 8  # Angle max (deg)
RATE_A = 9  # Thermal limit
B_ON = 10  # Branch on/off
