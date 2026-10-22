import { createElement, type SelectHTMLAttributes } from 'react';

/** Native selection with a shrinkable selected value in customizable browsers. */
export default function Select({ children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...props}>
      <button type="button" className="select-trigger">
        {createElement('selectedcontent', { className: 'select-value' })}
      </button>
      {children}
    </select>
  );
}
