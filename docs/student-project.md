# Guía del proyecto estudiantil

## Primera etapa: datos estáticos

1. instala el SDK y descarga el corte inicial;
2. valida continuidad, duplicados, tipos y cobertura por estación;
3. realiza análisis exploratorio temporal y geográfico;
4. construye al menos dos baselines;
5. usa backtesting temporal y conserva evidencia de cada experimento;
6. define cómo versionarás modelo, features y cutoff.

## Segunda etapa: operación incremental

Cuando se active el reloj, GitHub Actions deberá:

1. consultar únicamente observaciones nuevas;
2. persistir el cursor o último timestamp procesado;
3. calcular métricas y señales de drift;
4. decidir si conserva o reentrena el modelo;
5. generar los cuatro horizontes solicitados;
6. enviar la predicción con versión y commit;
7. registrar éxito o error de la ejecución.

## Entregables mínimos

- repositorio reproducible;
- README con arquitectura y decisiones;
- pipeline automático en GitHub Actions;
- validación temporal y comparación contra baselines;
- monitoreo de datos y desempeño;
- estrategia explícita de reentrenamiento;
- historial de predicciones y modelos.

## Bono de dashboard

El dashboard puede mostrar:

- serie y mapa por estación;
- distribución de errores;
- accuracy acumulada y rolling 24h;
- señales de data/concept drift;
- última ejecución del pipeline;
- versión activa del modelo;
- posición en el leaderboard.

Vercel y Supabase son opciones gratuitas recomendadas, no requisitos de la
métrica. Nunca expongas claves privadas de Supabase en el navegador.
