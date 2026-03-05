# FC-Proxy Deploy / Rollback

## Backup

- `tar -czf "/root/backup/fc-proxy_YYYYmmdd_HHMMSS.tgz" -C "/root" "fc-proxy"`

## Build & Restart

- `cd "/root/fc-proxy"`
- `docker compose build fc-proxy`
- `docker compose up -d fc-proxy`
- `docker logs fc-proxy --tail 200`

## Verify

- `curl "http://127.0.0.1:1030/health"`

## Rollback

- Stop current container: `docker compose down`
- Restore backup: `tar -xzf "/root/backup/fc-proxy_YYYYmmdd_HHMMSS.tgz" -C "/root"`
- Rebuild & restart: `docker compose build fc-proxy && docker compose up -d fc-proxy`

