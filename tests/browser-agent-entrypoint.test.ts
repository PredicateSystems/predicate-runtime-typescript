import fs from 'fs';
import path from 'path';

import * as browserAgent from '../src/browser-agent';

describe('browser-agent entrypoint', () => {
  it('exports the planner-executor primitives needed by browser extensions', () => {
    expect(browserAgent.PlannerExecutorAgent).toBeDefined();
    expect(browserAgent.LLMProvider).toBeDefined();
    expect(browserAgent.ConfigPreset).toBeDefined();
    expect(browserAgent.mergeConfig).toBeDefined();
    expect(browserAgent.parseAction).toBeDefined();
  });

  it('does not export the Playwright runtime from the browser-safe entrypoint', () => {
    expect('PlaywrightRuntime' in browserAgent).toBe(false);
    expect('createPlaywrightRuntime' in browserAgent).toBe(false);
  });

  it('declares a browser-agent package subpath export', () => {
    const packageJsonPath = path.join(__dirname, '..', 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8')) as {
      exports?: Record<string, unknown>;
    };

    expect(packageJson.exports?.['./browser-agent']).toEqual({
      types: './dist/browser-agent.d.ts',
      import: './dist/esm/browser-agent.js',
      require: './dist/browser-agent.js',
      default: './dist/browser-agent.js',
    });
  });
});
