# graphkit as a modular runtime for 3rd-party power-grid models

Design note. Goal: let an external team ship a power-grid model as a pip-installable
package (or a local folder) and run it through graphkit's training/eval/physics
stack **without forking graphkit**. Benchmarked against TerraTorch, which solves the
same problem for geospatial foundation models.

Status today: graphkit is ~70% there. It has decorator registries
(`io/registries.py`), config-driven selection (`args.model.type` → `load_model`),
and a manual `--plugins pkg` import hook. What's missing is (a) a *declared contract*
for what a plugged-in model must satisfy, (b) a decoupling seam so a 3rd party doesn't
have to inherit graphkit's hardcoded feature layout, and (c) frictionless discovery.

---

## What TerraTorch does that we should copy

TerraTorch does **not** register whole models. It registers *parts* — backbones, necks,
decoders — and an `EncoderDecoderFactory.build_model(...)` assembles them. Four things
transfer directly:

1. **Compose, don't monolith.** A 3rd party brings a *backbone*; they reuse our
   decoders, heads, physics, tasks. graphkit currently forces them to bring an entire
   `nn.Module` that already emits `{"bus","gen"}` in our exact column order.

2. **The neck = the decoupling layer.** TerraTorch necks "convert layer-wise backbone
   outputs into the required decoder inputs" — `list[Tensor] -> list[Tensor]`, with a
   parallel `process_channel_list` that tracks shapes. This is exactly the seam that
   quarantines our `globals.py` index contract: a 3rd-party backbone emits whatever it
   wants; an **adapter** maps it onto the `{bus,gen}` + column-index contract the tasks
   and physics expect. The coupling stops being global and becomes one registered object.

3. **`out_channels` as the backbone↔factory contract.** TerraTorch requires
   `backbone.out_channels: list[int]`; the factory reads it and threads it through necks
   to size the decoder. Our analogue: a backbone declares the feature schema it emits
   (node/edge types + per-type feature names), and the factory validates the adapter can
   bridge it to the task's required schema.

4. **`custom_modules_path` in the YAML.** TerraTorch's lightest extension path isn't an
   entry point or a CLI flag — it's a config key pointing at a folder, imported via
   `importlib` in a Lightning CLI hook so `@REGISTRY.register` decorators fire. The
   experiment is reproducible from the YAML alone. Strictly better than our `--plugins`
   flag.

One divergence to respect: TerraTorch's CV domain has a clean encoder/decoder split.
Power-grid GNNs entangle the graph schema (node/edge types) with physics. So our
"neck" carries more than channel counts — it carries the **graph schema** (which node
types, which per-type feature columns mean what).

---

## Current graphkit internals (verified)

- `Registry` (`io/registries.py:1`): `register(name)` decorator (name is *given*, not
  inferred), `get`, `create(name, *args)`, `__contains__`, `__iter__`. Seven singletons:
  `MODELS_REGISTRY`, `TASK_REGISTRY`, `LOSS_REGISTRY`, `NORMALIZERS_REGISTRY`,
  `TRANSFORM_REGISTRY`, `PHYSICS_DECODER_REGISTRY`, `DATASET_WRAPPER_REGISTRY`.
- Selection: `args.model.type` (YAML) → `load_model(args)` (`io/param_handler.py:121`)
  → `MODELS_REGISTRY.create(type, args)`. Model is a plain `nn.Module(args)`.
- Model↔task contract: PyG `HeteroData` with node types `bus`/`gen` and 3 relations;
  model returns `{"bus": Tensor, "gen": Tensor}`.
- **The coupling**: `datasets/globals.py` hardcodes feature *meaning* as integer indices
  (`VM_H=3`, `PG_OUT=2`, …), imported directly by both models and tasks. This is what a
  3rd party cannot escape today.
- Extension today: `--plugins pkg` (`cli.py:_load_plugins`) importlib-imports packages so
  their decorators fire. No entry-point group for graphkit's own registries. Config is
  `NestedNamespace` — schema-less `getattr`-with-defaults.

---

## Proposed design

### Tier 0 — formalize the contract (cheap, no behavior change)

Add `typing.Protocol`s so "what a plugin must satisfy" is declared, not folklore.

```python
# io/interfaces.py  (new)
class GridBackbone(Protocol):
    def __init__(self, args) -> None: ...
    def forward(self, batch) -> dict[str, Tensor]: ...   # native output, any schema
    @property
    def output_schema(self) -> "GraphSchema": ...        # what it emits

class GridAdapter(Protocol):                             # the "neck"
    def __init__(self, input_schema: "GraphSchema", target_schema: "GraphSchema"): ...
    def forward(self, feats: dict[str, Tensor]) -> dict[str, Tensor]: ...
```

`GraphSchema` is the declarative replacement for `globals.py`: node types, and per-type
an **ordered list of named feature columns** (so `VM_H` becomes
`schema["bus"].index("vm")` instead of a magic `3`). Ship the built-in layout as the
default `GraphSchema` so existing models keep working untouched.

This is additive: `GNS_heterogeneous` and `GRIT` already satisfy `GridBackbone`
structurally. The Protocol just documents and type-checks it.

### Tier 1 — the factory + the adapter seam

Add `GridModelFactory` mirroring `EncoderDecoderFactory`, selected by a new optional
config block. Backwards-compatible: if `model.type` names a whole model, behave exactly
as today; if it names a *backbone* + `model.adapter`, compose.

```python
# io/model_factory.py (new)
def build_model(args) -> nn.Module:
    if _is_composed(args):                          # new path
        backbone = MODELS_REGISTRY.create(args.model.backbone, args)
        adapter  = ADAPTER_REGISTRY.create(
            args.model.adapter, backbone.output_schema, task_schema(args))
        return ComposedGridModel(backbone, adapter) # forward: adapter(backbone(batch))
    return MODELS_REGISTRY.create(args.model.type, args)   # legacy path, unchanged
```

Add one registry: `ADAPTER_REGISTRY = Registry("adapter")`. The adapter is where the
`globals.py` index knowledge now lives — a 3rd party either reuses a built-in adapter
(their backbone emits our schema) or ships their own (theirs differs). Tasks/physics keep
consuming the fixed contract; they never see the plugin's native schema.

This reframes the hard problem: we are **not** rewriting every `globals.py` reference.
We are adding a quarantine boundary and pushing the index knowledge behind it.

### Tier 2 — frictionless discovery (three tiers, like TerraTorch)

1. `custom_modules_path:` YAML key → `importlib.import_module` in the CLI boot
   (add to `cli.py` next to `_load_plugins`). Reproducible from config alone. **Primary.**
2. `[project.entry-points."gridfm_graphkit.models"]` auto-scanned at startup — the
   pip-install-and-go tier for distributed packages. The team already ships vLLM
   entry-points, so the idiom is known.
3. Keep `--plugins pkg` (CLI) and allow passing a raw `nn.Module` for one-off experiments
   that don't warrant registration (TerraTorch allows this too).

### Tier 3 — config validation (optional, later)

`NestedNamespace` is schema-less. A plugin ecosystem eventually wants each backbone/adapter
to declare + validate its own config sub-block (pydantic or a lightweight schema on the
Protocol). Defer until there are real 3rd-party plugins to pressure-test it.

---

## Sequencing & risk

| Step | Effort | Risk | Unlocks |
|------|--------|------|---------|
| T0 Protocols + `GraphSchema` default | low | none (additive) | declared contract, typed plugins |
| T2.1 `custom_modules_path` | low | none | reproducible local extension |
| T2.2 entry-point discovery | low | none | pip-install-and-go |
| T1 factory + adapter seam | **med** | med (touches model-build path; gate behind new config) | backbones reuse our stack; `globals.py` quarantined |
| T3 config schema | med | low | per-plugin validation |

Do T0 + T2.1 + T2.2 first — they make graphkit a real plugin runtime for models that fit
today's grid schema, with zero behavior change. T1 is the one that makes it
*model-agnostic* rather than *GNS/GRIT-agnostic*; land it behind the new composed-model
config block so the legacy path is untouched.

Open question for T1: how much graph-schema generality do we actually need? If every
realistic 3rd-party model still speaks `bus`/`gen` + 3 relations, the adapter only
remaps feature columns (easy). If they bring new node/edge types, the physics decoders
and metrics need the schema too — bigger. Worth deciding the target before building T1.

---

## Implemented (this branch)

Two of the additive tiers are now live in graphkit; both are backward-compatible
(existing configs behave exactly as before).

**Entry-point auto-discovery (T2.2).** `cli.py` gains `_load_entrypoint_plugins()`,
which scans the `gridfm_graphkit.plugins` entry-point group at startup and loads each
(callable → called, module → imported). Run alongside `_load_plugins` at both CLI
entry sites. Effect: a pip-installed plugin registers itself with **no `--plugins`
flag**; `--plugins` still works for packages that aren't installed as distributions.
A failing plugin warns rather than aborting the CLI.

**Config-declared transform lists with task-name fallback.** `get_task_transforms`
(`io/param_handler.py`) now checks `data.transforms` first:

```yaml
data:
  transforms:
    - RemoveInactiveBranches                       # bare name
    - name: AddPFHeteroMask                        # name + per-transform kwargs
      mask_ref_vm: true
    - ApplyMasking
```

If present, it composes exactly those transforms in order, resolving each from
`TRANSFORM_REGISTRY` (built-in *or* plugin-registered). If absent, it falls back to the
historical single composition keyed by `task.task_name` — so every current config is
untouched. The individual built-in transforms are now registered under their class
names (`task_transforms.py`) so they can be referenced in such a list; `_build_transform`
handles their non-uniform constructors via a `(args, **kwargs)` → `(**kwargs)` →
`(args)` → `()` call convention.

Still not done here: T0 (`GraphSchema` + Protocols), T1 (adapter/neck + `GridModelFactory`),
T2.1 (`custom_modules_path`), T3 (config schema validation). The hello-world template at
`gridfm/gridfm-graphkit-module-template` exercises the two implemented tiers.
