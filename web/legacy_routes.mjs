// Historical HTML bookmarks retain the project prefix on GitHub Pages.
export function legacyDestination(pathname) {
    const match=pathname.match(/^(.*?)\/(v4\/v2|v4|web\/eye|web)(?:\/(.*))?$/);
    if(!match)return null;
    const [,root,old,rest='']=match;
    if(old==='web' && (!rest || rest==='index.html'))return root+'/index.html';
    const version=old==='v4/v2'||old==='web/eye'?'v2':'v1';
    if(!rest || rest==='index.html')return root+`/web/${version}/sky_atlas.html`;
    if(['sky_atlas.html','star_map.html'].includes(rest))return root+`/web/${version}/`+rest;
    if(old==='v4' && /^(data|design)\/.+\.md$/.test(rest))return root+'/'+rest;
    if((old==='v4/v2'||old==='web/eye') && rest==='tests/render_checks.html')return root+'/web/v2/'+rest;
    return null;
}
