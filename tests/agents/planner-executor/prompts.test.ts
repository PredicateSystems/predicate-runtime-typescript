import { buildExecutorPrompt } from '../../../src/agents/planner-executor/prompts';

describe('buildExecutorPrompt', () => {
  it('accepts broader search input labels for TYPE_AND_SUBMIT actions', () => {
    const [systemPrompt] = buildExecutorPrompt(
      'searchbox',
      'searchbox',
      '168|searchbox|Search Amazon|1530|1||1|Search Amazon|1|1|',
      'noise cancelling earbuds',
      'shopping',
      'TYPE_AND_SUBMIT'
    );

    expect(systemPrompt).toContain('Search Amazon');
    expect(systemPrompt).toContain('Search products');
    expect(systemPrompt).not.toContain(
      "ONLY use fields explicitly labeled for SEARCH (placeholder='Search', aria='Search')."
    );
  });

  it('uses strong target-text matching rules when a click action has planner-supplied text', () => {
    const [systemPrompt] = buildExecutorPrompt(
      'product link',
      'product link',
      '968|link|JBL Vibe Beam 2 - True Wireless Earbuds|1200|1||1|JBL audio|1|1|/dp/example',
      'JBL Vibe Beam 2 - True Wireless...',
      'shopping',
      'CLICK'
    );

    expect(systemPrompt).toContain('CRITICAL: Find an element with text matching');
    expect(systemPrompt).toContain(
      'If multiple elements match, choose the one with the strongest text overlap'
    );
    expect(systemPrompt).toContain('If NO element contains the target text, return NONE');
  });
});
