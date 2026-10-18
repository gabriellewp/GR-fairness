"""Source installation for the FairDiverse generative fairness release."""
from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="fairdiverse",
    version="1.0.0",
    description="Fairness and diversity experiments with GRID and classic recommenders",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(include=["fairdiverse", "fairdiverse.*", "additional_baselines", "additional_baselines.*"],
                           exclude=["*.tests", "*.tests.*"]),
    package_data={"fairdiverse": ["recommendation/*.yaml", "recommendation/properties/*.yaml",
                                 "recommendation/properties/dataset/*.yaml", "recommendation/properties/models/*.yaml",
                                 "search/*.yaml", "search/properties/*.yaml"],
                  "additional_baselines": ["configs/*.json"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.24,<3", "torch>=2.4,<3", "PyYAML>=6", "pandas>=2,<3",
                      "scipy>=1.10", "scikit-learn>=1.3", "tqdm>=4.65", "protobuf>=4.25"],
    extras_require={
        "grid": ["transformers==4.57.6", "lightning>=2.4,<2.6", "torchmetrics>=1.3,<2",
                 "hydra-core>=1.3,<1.4", "omegaconf>=2.3,<2.4", "rootutils>=1.0", "rich>=13",
                 "pyarrow>=14", "tensorflow-cpu>=2.16,<3", "python-dotenv>=1", "psutil>=5",
                 "sentencepiece>=0.2", "wandb>=0.17"],
        "rerank": ["cvxpy>=1.6,<2", "POT>=0.9", "PuLP>=2.8"],
        "analysis": ["ijson>=3.2", "matplotlib>=3.8"],
        "test": ["pytest>=8,<9"],
    },
    zip_safe=False,
)
