# Place ERP dumps here (gitignored). Expected default:
#   data/dumps/pdv_prod.dump   (pg_dump -Fc)
# or
#   data/dumps/pdv_prod.sql    (plain SQL) — set ERP_DUMP_PATH accordingly
#
# Generate from the ERP host (example):
#   pg_dump -Fc -h HOST -U USER -d pdv_prod -f data/dumps/pdv_prod.dump
#
# Then:
#   docker compose up -d postgres-erp
#   uv run restore-erp-dump
#   uv run introspect-erp
