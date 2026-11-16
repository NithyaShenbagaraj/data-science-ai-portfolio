# Data directory

- `raw/`: source extracts; never commit private or large raw datasets.
- `interim/`: intermediate cleaned datasets.
- `processed/`: model-ready datasets.
- `data_dictionary.md`: planned canonical field definitions.

Agency contracts, retainers, invoices, payments and client renewal histories will initially be synthetic. Label generated rows with a source/provenance field and document generator assumptions. Public campaign datasets must retain their source, licence and attribution details.

Do not mix agency-recognized revenue with client-attributed campaign revenue. Record currency, timezone, event date, recognition date and reporting grain explicitly.
