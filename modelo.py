"""Modelo térmico de parámetros concentrados (1 nodo) de un aula.

Balance de energía sobre el aire + masa térmica interior:

    C · dT/dt = Q_personas + Q_equipos + Q_solar
                + UA·(T_ext - T)                 (envolvente)
                + ACH·V·ρ·cp/3600·(T_ext - T)    (ventilación / infiltración)
                - Q_aire                         (aire acondicionado)

Se integra con Euler explícito, paso de 60 s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

RHO_AIRE = 1.2          # kg/m³
CP_AIRE = 1005.0        # J/(kg·K)
FRIGORIA_A_W = 1.163    # 1 frigoría/h = 1,163 W

# Capacidad térmica efectiva por m² de piso (kJ/(m²·K)): paredes interiores, mobiliario, etc.
MASA_TERMICA = {
    "Liviana (placas, steel frame)": 80.0,
    "Media (ladrillo hueco)": 165.0,
    "Pesada (hormigón / macizo)": 260.0,
}


@dataclass
class Parametros:
    # --- Geometría ---
    largo: float = 8.0
    ancho: float = 7.5
    alto: float = 3.0
    masa_termica_kj_m2k: float = 165.0

    # --- Envolvente ---
    n_paredes_ext: int = 1
    frac_vidrio: float = 0.25        # fracción de la pared exterior que es ventana
    u_muro: float = 1.8              # W/(m²·K)
    u_vidrio: float = 5.8            # W/(m²·K)
    techo_expuesto: bool = True
    u_techo: float = 2.5             # W/(m²·K)
    factor_solar: float = 0.6        # fracción de la radiación que entra (cortinas, aleros)

    # --- Ocupación y cargas internas ---
    n_alumnos: int = 30
    q_persona: float = 75.0          # W de calor sensible por persona
    w_equipos: float = 300.0         # W (proyector, notebook, etc.)
    w_luz_m2: float = 10.0           # W/m² de iluminación
    h_ingreso: float = 8.0
    h_salida: float = 13.0

    # --- Clima exterior ---
    t_ext_min: float = 21.0
    t_ext_max: float = 33.0
    irr_max: float = 600.0           # W/m² sobre la fachada al mediodía

    # --- Ventilación ---
    ach_cerrado: float = 0.5         # renovaciones/hora con ventanas cerradas
    ach_abierto: float = 4.0         # renovaciones/hora con ventanas abiertas
    ventanas_abiertas: bool = True   # solo cuando el aire NO está funcionando

    # --- Aire acondicionado ---
    frigorias: float = 6000.0
    t_consigna: float = 24.0
    modo: str = "Inverter"           # "Inverter" o "On/Off"
    banda: float = 0.5               # °C (histéresis On/Off; banda proporcional Inverter)
    cop: float = 3.0
    h_on: float = 7.5
    h_off: float = 13.5

    # --- Simulación ---
    h_ini: float = 6.0
    h_fin: float = 18.0
    t_inicial: float = 25.0

    # ---- Derivados ----
    @property
    def area(self) -> float:
        return self.largo * self.ancho

    @property
    def volumen(self) -> float:
        return self.area * self.alto

    @property
    def area_pared_ext(self) -> float:
        return self.n_paredes_ext * self.largo * self.alto

    @property
    def area_vidrio(self) -> float:
        return self.frac_vidrio * self.area_pared_ext

    @property
    def ua(self) -> float:
        """Conductancia global de la envolvente (W/K)."""
        a_muro = self.area_pared_ext - self.area_vidrio
        ua = a_muro * self.u_muro + self.area_vidrio * self.u_vidrio
        if self.techo_expuesto:
            ua += self.area * self.u_techo
        return ua

    @property
    def capacidad(self) -> float:
        """Capacidad térmica total (J/K): aire + masa interior."""
        c_aire = RHO_AIRE * CP_AIRE * self.volumen
        c_masa = self.masa_termica_kj_m2k * 1000.0 * self.area
        return c_aire + c_masa


def t_exterior(h: float, p: Parametros) -> float:
    """Temperatura exterior sinusoidal: mínima ~3 h, máxima ~15 h."""
    media = (p.t_ext_min + p.t_ext_max) / 2
    amp = (p.t_ext_max - p.t_ext_min) / 2
    return media + amp * math.sin(2 * math.pi * (h - 9) / 24)


def irradiancia(h: float, p: Parametros) -> float:
    """Radiación sobre la fachada: 0 de noche, pico al mediodía solar (12 h)."""
    return p.irr_max * max(0.0, math.sin(math.pi * (h - 6) / 12))


def simular(p: Parametros, con_aire: bool) -> pd.DataFrame:
    dt = 60.0  # s
    horas = np.arange(p.h_ini, p.h_fin + 1e-9, dt / 3600)
    n = len(horas)

    claves = ["t_int", "t_ext", "personas", "q_personas", "q_equipos", "q_solar",
              "q_envolvente", "q_ventilacion", "q_aire", "p_electrica"]
    r = {k: np.zeros(n) for k in claves}

    q_nom = p.frigorias * FRIGORIA_A_W
    C = p.capacidad
    T = p.t_inicial
    compresor = False

    for i, h in enumerate(horas):
        clase = p.h_ingreso <= h < p.h_salida
        t_ext = t_exterior(h, p)

        presentes = p.n_alumnos if clase else 0
        q_pers = presentes * p.q_persona
        q_equ = (p.w_equipos + p.w_luz_m2 * p.area) if clase else 0.0
        q_sol = p.area_vidrio * irradiancia(h, p) * p.factor_solar
        q_env = p.ua * (t_ext - T)

        ac_programado = con_aire and (p.h_on <= h < p.h_off)
        # Con el aire encendido se asumen ventanas cerradas
        abiertas = p.ventanas_abiertas and clase and not ac_programado
        ach = p.ach_abierto if abiertas else p.ach_cerrado
        q_vent = ach * p.volumen * RHO_AIRE * CP_AIRE / 3600.0 * (t_ext - T)

        q_ac = 0.0
        if ac_programado:
            if p.modo == "Inverter":
                q_ac = q_nom * float(np.clip((T - p.t_consigna) / p.banda, 0.0, 1.0))
            else:
                if T > p.t_consigna + p.banda:
                    compresor = True
                elif T < p.t_consigna - p.banda:
                    compresor = False
                q_ac = q_nom if compresor else 0.0
        else:
            compresor = False

        r["t_int"][i] = T
        r["t_ext"][i] = t_ext
        r["personas"][i] = presentes
        r["q_personas"][i] = q_pers
        r["q_equipos"][i] = q_equ
        r["q_solar"][i] = q_sol
        r["q_envolvente"][i] = q_env
        r["q_ventilacion"][i] = q_vent
        r["q_aire"][i] = q_ac
        r["p_electrica"][i] = q_ac / p.cop

        T += (q_pers + q_equ + q_sol + q_env + q_vent - q_ac) * dt / C

    df = pd.DataFrame({"hora": horas, **r})
    df["kwh"] = np.cumsum(df["p_electrica"]) * dt / 3.6e6
    return df


def resumen(df: pd.DataFrame, p: Parametros, t_confort: float) -> dict:
    """Indicadores clave de una corrida."""
    dt_h = (df["hora"].iloc[1] - df["hora"].iloc[0])
    clase = (df["hora"] >= p.h_ingreso) & (df["hora"] < p.h_salida)
    en_clase = df[clase]
    return {
        "t_max": float(df["t_int"].max()),
        "t_prom_clase": float(en_clase["t_int"].mean()) if len(en_clase) else float("nan"),
        "h_sobre_confort": float((en_clase["t_int"] > t_confort).sum() * dt_h),
        "kwh": float(df["kwh"].iloc[-1]),
    }
