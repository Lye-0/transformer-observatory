import assert from 'node:assert/strict';
import test from 'node:test';
import { tokenLabel, tokenTitle } from '../src/tokenText.ts';

test('Japanese and emoji use decoded text while raw token bytes and IDs remain inspectable', () => {
  const token = { id: 89015, text: 'こんにちは', raw: 'ãģĵãĤĵãģ«ãģ¡ãģ¯' };
  assert.equal(tokenLabel(token), 'こんにちは');
  assert.ok(tokenTitle(token).includes(token.raw));
  assert.ok(tokenTitle(token).includes('89015'));
  assert.equal(tokenLabel({ text: '🌌', raw: 'byte-encoded' }), '🌌');
});

test('spaces, line breaks, tabs and control characters are visible, without changing the token data', () => {
  const token = { text: ' sky\n\t\r\0\x7f', raw: 'raw' };
  assert.equal(tokenLabel(token), '␠sky↵⇥\\u000d\\u0000\\u007f');
  assert.equal(token.text, ' sky\n\t\r\0\x7f');
});

test('empty special tokens and incomplete byte fragments preserve raw representation', () => {
  assert.equal(tokenLabel({ text: '', raw: '<BOS>' }), '<BOS>');
  assert.equal(tokenLabel({ text: '�', raw: '<E3>' }), '<E3>');
  assert.ok(tokenTitle({ text: '�', raw: '<E3>' }).includes('置換文字'));
  assert.equal(tokenLabel(undefined), '');
  const legacy = { text: 17, raw: '<17>' } as unknown as { text: string; raw: string };
  assert.equal(tokenLabel(legacy), '<17>');
  assert.doesNotThrow(() => tokenTitle(legacy));
});
