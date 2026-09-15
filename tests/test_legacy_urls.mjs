import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {legacyDestination} from '../web/legacy_routes.mjs';

test('working tree no longer has a v4 directory', () => {
    assert.equal(existsSync(new URL('../v4', import.meta.url)), false);
});

test('root 404 page remaps historical /v4/ bookmarks', async () => {
    const html = await readFile(new URL('../404.html', import.meta.url), 'utf8');
    assert.ok(html.includes("import(root+'/web/legacy_routes.mjs')"));
    assert.ok(html.includes('destination+location.search+location.hash'));
    assert.equal(legacyDestination('/v4'), '/web/v1/sky_atlas.html');
    assert.equal(legacyDestination('/v4/'), '/web/v1/sky_atlas.html');
    assert.equal(legacyDestination('/v4/sky_atlas.html'), '/web/v1/sky_atlas.html');
    assert.equal(legacyDestination('/v4/star_map.html'), '/web/v1/star_map.html');
    assert.equal(legacyDestination('/worldbuilding/v4/sky_atlas.html'), '/worldbuilding/web/v1/sky_atlas.html');
    assert.equal(legacyDestination('/v4/v2'), '/web/v2/sky_atlas.html');
    assert.equal(legacyDestination('/v4/v2/'), '/web/v2/sky_atlas.html');
    assert.equal(legacyDestination('/v4/v2/star_map.html'), '/web/v2/star_map.html');
    assert.equal(legacyDestination('/worldbuilding/v4/v2/sky_atlas.html'), '/worldbuilding/web/v2/sky_atlas.html');
    assert.equal(legacyDestination('/web/sky_atlas.html'), '/web/v1/sky_atlas.html');
    assert.equal(legacyDestination('/web/eye/star_map.html'), '/web/v2/star_map.html');
    assert.equal(legacyDestination('/web/'), '/index.html');
    assert.equal(legacyDestination('/v4/design/terrax_precession_nutation.md'), '/design/terrax_precession_nutation.md');
    assert.equal(legacyDestination('/web/v1/sky_atlas.html'), null);
    assert.equal(legacyDestination('/web/v2/star_map.html'), null);
});
