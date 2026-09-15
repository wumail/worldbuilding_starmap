// Lines use an absolute opacity, matching the former blue/green reference
// circles (.5 * .42). Do not multiply already dim grid colors/alphas again.
export const SKY_OVERLAYS=Object.freeze({line:.21,text:.48,marker:.45,border:.65});
if(typeof document!=='undefined')document.documentElement.style.setProperty('--sky-overlay-text',String(SKY_OVERLAYS.text));
