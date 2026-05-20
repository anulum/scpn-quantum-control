# Paper 0 Front Matter Context Specs

- Source span: P0R00018 - P0R00104
- Source records: 87
- Consumed source records: 87
- Coverage match: True
- Spec count: 5
- Collection books: 5
- Layer monographs: 16
- Validation suite papers: 4
- Blank ToC placeholders: 45
- Fragmented ToC warning present: True
- Hardware status: source_context_no_experiment
- Claim boundary: source-bounded front matter context; not validation evidence

## Specs
- `front_matter_context.collection_identity`: The front matter identifies author, collection position, Book II location, and the hypothesis/falsifiability status of Paper 0.
- `front_matter_context.master_publication_topology`: The master publication list maps Paper 0, the 16 layer monographs, and Papers 17-20 validation and synthesis suite.
- `front_matter_context.chapter_structure_marker`: The source marks the local Paper 0 chapter-structure table before the blank placeholder block.
- `front_matter_context.blank_toc_placeholders`: The canonical ledger contains 45 blank ToC placeholder records that must remain counted rather than silently skipped.
- `front_matter_context.fragmented_toc_warning`: The source explicitly warns that the ToC is fragmented and currently incorrect.
