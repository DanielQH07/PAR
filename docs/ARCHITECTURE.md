# Repository and experiment architecture

This repository separates source code, runtime data, configuration, tests, and experiment records. Source of the diagrams is in [`repository-architecture.mmd`](repository-architecture.mmd) and [`experiment-flow.mmd`](experiment-flow.mmd).

## Repository layout

```mermaid
flowchart TB
  classDef app fill:#172554,stroke:#7dd3fc,color:#f8fafc,stroke-width:1.5px
  classDef data fill:#14532d,stroke:#86efac,color:#f0fdf4,stroke-width:1.5px
  classDef docs fill:#3f1d3a,stroke:#f0abfc,color:#fdf4ff,stroke-width:1.5px

  subgraph ROOT["Repository root"]
    CONFIG["config/<br/>YAML run configs"]
    DATA["data/<br/>downloaded images and caches<br/>ignored by Git"]
    TESTS["tests/<br/>layout and metric checks"]
    EXP["experiments/<br/>protocols, reports, run metadata"]
    DOCS["docs/ and Markdown<br/>runbooks and diagrams"]
  end

  subgraph SRC["src/"]
    ENTRY["train_*.py, infer_*.py<br/>feature and evaluation scripts"]
    PYCONFIG["configs/<br/>Python config loader"]
    DATASET["dataset/<br/>loaders and transforms"]
    MODELS["models/<br/>backbones and heads"]
    subgraph UTILS["utils/"]
      LOSS["losses/"]
      METRIC["metrics/"]
      OPTIM["optim/"]
      TOOLS["tools/<br/>common helpers"]
    end
  end

  CONFIG --> ENTRY
  DATA --> DATASET
  ENTRY --> DATASET
  ENTRY --> MODELS
  ENTRY --> UTILS
  ENTRY --> EXP
  TESTS --> SRC

  class CONFIG,TESTS,EXP,ENTRY,PYCONFIG,DATASET,MODELS,LOSS,METRIC,OPTIM,TOOLS app
  class DATA data
  class DOCS docs
```

`config/` contains user-editable YAML files. `src/configs/` is the Python package that parses them. `src/utils/` contains the moved loss, metric, optimizer, and helper packages. Run Python entry points from the repository root as `python src/<script>.py`; the script directory then supplies the import root for `configs`, `dataset`, `models`, and `utils`.

## Data and training flow

```mermaid
flowchart LR
  classDef input fill:#172554,stroke:#7dd3fc,color:#f8fafc,stroke-width:1.5px
  classDef process fill:#14532d,stroke:#86efac,color:#f0fdf4,stroke-width:1.5px
  classDef output fill:#3f1d3a,stroke:#f0abfc,color:#fdf4ff,stroke-width:1.5px

  IMAGES["Official images<br/>Market1501, PA-100K, PETA"]
  LABELS["Official train/val CSVs<br/>40 binary attributes"]
  EXTRACT["src/extract_2027_backbone_features.py<br/>frozen backbone + AMP"]
  CACHE["data/feature_cache_<br/>float32 features and labels"]
  PROBE["src/train_linear_probe_2027.py<br/>LayerNorm + Linear head"]
  PARTIAL["src/train_partial_ft_2027.py<br/>optional top-block fine-tune"]
  EVAL["Validation and LODO<br/>threshold 0.5"]
  REPORT["experiments/results/<br/>metrics, per-attribute CSV, configs"]

  IMAGES --> EXTRACT
  LABELS --> EXTRACT
  EXTRACT --> CACHE
  CACHE --> PROBE
  CACHE --> EVAL
  PROBE --> EVAL
  IMAGES --> PARTIAL
  LABELS --> PARTIAL
  PARTIAL --> EVAL
  EVAL --> REPORT

  class IMAGES,LABELS input
  class EXTRACT,CACHE,PROBE,PARTIAL,EVAL process
  class REPORT output
```

For setup commands and model/data links, see [`RUN_RESEARCH.md`](../RUN_RESEARCH.md). Training protocols and current score limitations are described in [`../experiments/README.md`](../experiments/README.md).
