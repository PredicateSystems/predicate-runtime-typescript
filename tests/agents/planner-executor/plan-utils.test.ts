import { parseAction } from '../../../src/agents/planner-executor';

describe('parseAction', () => {
  it('parses clean single-line action outputs', () => {
    expect(parseAction('CLICK(42)')).toEqual({ action: 'CLICK', args: [42] });
    expect(parseAction('TYPE(7, "hello world")')).toEqual({
      action: 'TYPE',
      args: [7, 'hello world'],
    });
    expect(parseAction('NONE')).toEqual({ action: 'NONE', args: [] });
  });

  it('parses a standalone final action line while ignoring reasoning examples', () => {
    expect(
      parseAction(
        [
          'I considered the example TYPE(42, "hello world"), but that is only illustrative.',
          'Final action:',
          'CLICK(91)',
        ].join('\n')
      )
    ).toEqual({
      action: 'CLICK',
      args: [91],
    });
  });

  it('parses markdown-wrapped action lines', () => {
    expect(
      parseAction(['Here is the action:', '```', 'TYPE(7, "hello world")', '```'].join('\n'))
    ).toEqual({
      action: 'TYPE',
      args: [7, 'hello world'],
    });
  });

  it('does not treat action examples inside prose as executable output', () => {
    expect(
      parseAction(
        'The example output is TYPE(42, "hello world"), so we should not use that here. Therefore return NONE.'
      )
    ).toEqual({
      action: 'UNKNOWN',
      args: [
        'The example output is TYPE(42, "hello world"), so we should not use that here. Therefore return NONE.',
      ],
    });

    expect(parseAction('I think CLICK(12) would work, but I am not sure.')).toEqual({
      action: 'UNKNOWN',
      args: ['I think CLICK(12) would work, but I am not sure.'],
    });
  });
});
