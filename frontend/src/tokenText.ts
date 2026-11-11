type TokenText = { text?: string; raw?: string; id?: number } | null | undefined;

/** Display decoded text without hiding whitespace or inventing characters for byte fragments. */
export function tokenLabel(token: TokenText): string {
  const decoded = token?.text;
  const text = typeof decoded === 'string' ? decoded : '';
  const raw = typeof token?.raw === 'string' ? token.raw : '';
  const value = text && !text.includes('\uFFFD') ? text : raw || text;
  return value.replace(/[\u0000-\u0020\u007f]/g, (char) => {
    if (char === ' ') return '␠';
    if (char === '\n') return '↵';
    if (char === '\t') return '⇥';
    return '\\u' + char.charCodeAt(0).toString(16).padStart(4, '0');
  });
}

export function tokenTitle(token: TokenText): string {
  const lines = [`表示: ${tokenLabel(token)}`, `Token ID: ${token?.id ?? '—'}`, `raw: ${token?.raw ?? ''}`];
  if (typeof token?.text === 'string' && token.text.includes('\uFFFD')) {
    lines.push('単体のデコードに置換文字が含まれるため、raw表記を表示しています。');
  }
  return lines.join('\n');
}
