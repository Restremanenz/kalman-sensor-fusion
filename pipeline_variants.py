from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineOptions:
    """Aktive Korrekturen einer reproduzierbaren Pipelinevariante."""

    name: str
    label: str
    use_barometer: bool
    use_zupt: bool
    use_endpoint: bool
    use_rts: bool
    use_wall_constraint: bool
    use_lateral_corridor: bool
    use_video_in_filter: bool = False

    def __post_init__(self):
        if self.use_endpoint and not self.use_rts:
            raise ValueError(
                f"Pipelinevariante {self.name}: Eine Endpunktbedingung "
                "benötigt den RTS-Smoother."
            )


PIPELINE_VARIANTS = {
    "V1_IMU": PipelineOptions(
        name="V1_IMU",
        label="Reine inertiale Rekonstruktion",
        use_barometer=False,
        use_zupt=False,
        use_endpoint=False,
        use_rts=False,
        use_wall_constraint=False,
        use_lateral_corridor=False,
    ),
    "V2_BARO": PipelineOptions(
        name="V2_BARO",
        label="ESKF + Barometer",
        use_barometer=True,
        use_zupt=True,
        use_endpoint=False,
        use_rts=False,
        use_wall_constraint=False,
        use_lateral_corridor=False,
    ),
    "V3_RTS": PipelineOptions(
        name="V3_RTS",
        label="ESKF + Barometer + Endpunkt + RTS",
        use_barometer=True,
        use_zupt=True,
        use_endpoint=True,
        use_rts=True,
        use_wall_constraint=False,
        use_lateral_corridor=False,
    ),
    "V4_WALL": PipelineOptions(
        name="V4_WALL",
        label="V3 + kontinuierliches Wand-Constraint",
        use_barometer=True,
        use_zupt=True,
        use_endpoint=True,
        use_rts=True,
        use_wall_constraint=True,
        use_lateral_corridor=False,
    ),
    "V5_CURRENT": PipelineOptions(
        name="V5_CURRENT",
        label="V4 + lateraler Korridor",
        use_barometer=True,
        use_zupt=True,
        use_endpoint=True,
        use_rts=True,
        use_wall_constraint=True,
        use_lateral_corridor=True,
    ),
}


def get_pipeline_options(name):
    """Gibt eine unveränderliche Pipelinekonfiguration anhand ihres Namens zurück."""
    try:
        return PIPELINE_VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(PIPELINE_VARIANTS)
        raise ValueError(
            f"Unbekannte Pipelinevariante '{name}'. Verfügbar: {available}"
        ) from exc
