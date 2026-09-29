// Builds an element. `props` sets attributes, except `className`, `text` and
// event handlers (`onclick` and the like), which set the property of that name.
// Text is always set as text, never parsed as HTML.
export function h(tag, props = {}, ...children) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === undefined || value === false) continue;
    if (name === 'text') element.textContent = value;
    else if (name === 'className' || name.startsWith('on')) element[name] = value;
    else element.setAttribute(name, value === true ? '' : value);
  }
  element.append(...children.flat().filter((child) => child != null));
  return element;
}
