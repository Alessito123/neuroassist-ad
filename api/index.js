export default function handler(request, response) {
  const target = process.env.STREAMLIT_APP_URL;
  if (target) {
    response.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate=300");
    response.redirect(307, target);
    return;
  }

  response.status(200).setHeader("Content-Type", "text/html; charset=utf-8");
  response.end(`<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroAssist AD</title><style>
body{margin:0;background:#f7fafc;color:#173b57;font-family:Inter,system-ui,sans-serif;display:grid;min-height:100vh;place-items:center}
main{max-width:720px;margin:2rem;padding:2.5rem;border-radius:20px;background:white;box-shadow:0 18px 55px #173b5720}
.tag{display:inline-block;padding:.35rem .7rem;border-radius:999px;background:#eaf2f5;color:#1e6f8c;font-weight:700}
h1{font-size:2.5rem;margin:.8rem 0}.warn{border-left:4px solid #d99122;padding:.8rem 1rem;background:#fff8eb}code{background:#eef3f6;padding:.15rem .35rem;border-radius:4px}
</style></head><body><main><span class="tag">Vercel gateway</span><h1>🧠 NeuroAssist AD</h1>
<p>El gateway está desplegado correctamente. Falta configurar <code>STREAMLIT_APP_URL</code> con la URL del servicio persistente que ejecuta Streamlit.</p>
<p class="warn"><strong>Aviso:</strong> herramienta educativa de apoyo; no constituye un diagnóstico médico.</p>
<p>Estado técnico: <a href="/health">/health</a></p></main></body></html>`);
}

