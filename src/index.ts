import { Router } from 'itty-router';

const router = Router();

// Durable Object for container management
export class WhisperContainer {
  state: DurableObjectState;
  container: any;

  constructor(state: DurableObjectState) {
    this.state = state;
  }

  async fetch(request: Request) {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'ok',
        service: 'Whisper Container Worker',
        timestamp: new Date().toISOString()
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Ensure container is running
    if (!this.container) {
      this.container = await this.state.container.start();
    }

    // Proxy requests to the container
    try {
      const response = await this.container.fetch(request);
      return response;
    } catch (error) {
      return new Response(
        JSON.stringify({
          error: 'Failed to proxy to Whisper container',
          details: (error as Error).message,
        }),
        {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }
}

// Worker router for external requests
router.get('/health', () => ({
  status: 'ok',
  service: 'Whisper Container Worker',
  timestamp: new Date().toISOString()
}));

// Proxy all requests to the container via Durable Object
router.all('*', async (request: any) => {
  try {
    // Get the Durable Object
    const id = (globalThis as any).WHISPER_CONTAINER.idFromName('whisper-instance');
    const stub = (globalThis as any).WHISPER_CONTAINER.get(id);

    // Forward the request
    const response = await stub.fetch(request);
    return response;
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: 'Failed to proxy to Whisper container',
        details: (error as Error).message,
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
});

// Export for Cloudflare Workers
export default {
  async fetch(request: Request, env: any) {
    // Make Durable Object available globally
    (globalThis as any).WHISPER_CONTAINER = env.WHISPER_CONTAINER;
    return router.handle(request);
  },
};
