# License scope and artifact terms

## Integration package and original contributions

The FairDiverse integration package and original code, configuration,
documentation, and workflow contributions in this release are provided under
the [root MIT license](../LICENSE), except where separate third-party terms
apply. The package metadata in [setup.py](../setup.py) declares this MIT license.

The root license retains the existing `Copyright (c) 2020 RUCAIBox` notice from
upstream FairDiverse. Existing source attributions and license notices remain
in place. MIT redistribution requires retaining its copyright and permission
notice; the complete terms are in the license file.

## Bundled GRID and third-party components

The `GRID/` directory has a separate [Snap Inc. license](../GRID/LICENSE) permitting
non-commercial research use. It does not grant a commercial license. Redistributions
of GRID must retain or reproduce that license, including its copyright notices,
conditions, and disclaimers.

GRID's [notice index](../GRID/notices.txt) points to the preserved, ordered
third-party notices in `GRID/third_party_notices/`. Those notices and any separate
license notices in individual source files continue to apply to their respective
components. Installed dependencies also retain their own licenses.

The root MIT license does not relicense GRID or other third-party components.
When a workflow uses GRID, its separate terms apply to that component even though
the integration package declares MIT.

## Datasets, metadata, and checkpoints

The source release excludes datasets, embeddings, pretrained models, and
checkpoints. The root MIT license does not automatically cover those artifacts.
Preprocessing a dataset or training with this code does not by itself establish
permission to redistribute the resulting artifacts under MIT.

Any separately published artifact release should state its own applicable terms,
identify upstream data and model sources, and include required notices. Confirm
redistribution permission for copied metadata and other source content before
uploading processed data. Check applicable source, data, and model terms before
choosing a checkpoint license. Hosting on Hugging Face does not change those terms.

For anonymous review, retain required license and attribution notices. Check
artifact metadata and external download links for unintended personal information
without removing legally required notices.
