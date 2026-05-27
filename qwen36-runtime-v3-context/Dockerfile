FROM sz.cogiot.com/id348504946c01000/qwen36-27b-q4-runtime-with-model:20260427-v2

USER root
COPY runtime /app/runtime
COPY scripts /app/scripts
COPY container/entrypoint.sh /usr/local/bin/llm-runtime-entrypoint
RUN chmod +x /usr/local/bin/llm-runtime-entrypoint && \
    chmod +x /app/scripts/*.sh && \
    chmod +x /app/scripts/*.py
USER unsloth:runtimeusers

ENTRYPOINT ["/usr/local/bin/llm-runtime-entrypoint"]
