# Pulso TransMi — Dashboard

Bono opcional (`docs/student-project.md`): sitio estático (Vite + React +
TypeScript) que lee Supabase directo desde el navegador con la clave pública,
y una función serverless de Vercel que hace de proxy para el leaderboard de
la API de Pulso (esa clave sí es privada).

Ver [`docs/dashboard.md`](../docs/dashboard.md) en la raíz del repo para la
arquitectura completa y los pasos de despliegue en Vercel.

## Desarrollo local

```bash
cp .env.example .env   # completa con tus credenciales
npm install
npm run dev
```

## Build

```bash
npm run build
```
