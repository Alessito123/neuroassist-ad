export default function handler(request, response) {
  response.status(200).json({
    status: "ok",
    service: "neuroassist-ad-gateway",
    streamlitConfigured: Boolean(process.env.STREAMLIT_APP_URL),
    timestamp: new Date().toISOString()
  });
}

