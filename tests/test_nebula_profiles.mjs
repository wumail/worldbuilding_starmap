import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {MORPHOLOGY_MODEL,SOFT_MORPHOLOGY_MODEL,MORPHOLOGY_FAMILIES,morphologyWeight,morphologySolidAngle,gradientNoise,structure,validateMorphology,softShellColumn,filteredMorphologyGrid} from '../web/shared/nebula_morphology.mjs';
import {prepareDeepSources,profileValue} from '../web/shared/deep_sky_profiles.mjs';
const shape=family=>({model:MORPHOLOGY_MODEL,family,position_angle_rad:1.32,axis_ratio:.61,phase_rad:2.3,noise_seed:143,turbulence:.82,shell_thickness:.26,filament_width:.14});

test('soft shell line of sight agrees with independent adaptive emissivity integrals',()=>{
    const refs=JSON.parse(readFileSync(new URL('./fixtures/nebula_soft_shell_reference.json',import.meta.url)));
    for(const r of refs.samples)assert.ok(Math.abs(softShellColumn(r.impact,r.thickness)-r.column)<2e-6,JSON.stringify(r));
});

test('CPU JavaScript matches the Python exporter noise through lattice boundaries and negative coordinates',()=>{
    const refs=JSON.parse(readFileSync(new URL('./fixtures/nebula_noise_reference.json',import.meta.url)));
    for(const r of refs){
        assert.ok(Math.abs(gradientNoise(r.x,r.y,r.seed)-r.noise)<1e-10);
        assert.ok(Math.abs(structure(r.x,r.y,r.seed)-r.structure)<1e-10);
    }
});
test('browser and exporter reject incompatible cloud types and non-integer noise seeds',()=>{
    assert.throws(()=>validateMorphology(shape('shell'),'reflection_nebula'));
    assert.throws(()=>validateMorphology(shape('turbulent'),'open_cluster'));
    assert.throws(()=>validateMorphology({...shape('filament'),noise_seed:.5},'emission_nebula'));
});

test('non-radial luminosity normalization converges under independent Cartesian quadrature',()=>{
    // Independent rectangular midpoint sum, not the production polar grid.
    const cut=.9,t=.03,n=750,step=2*cut/n;
    for(const family of MORPHOLOGY_FAMILIES){
        const m=shape(family);let sum=0;
        for(let j=0;j<n;j++)for(let i=0;i<n;i++){
            const x=-cut+(i+.5)*step,y=-cut+(j+.5)*step;
            sum+=morphologyWeight(x,y,'ionized_gaussian',cut,m,profileValue)/(1+t*t*(x*x+y*y))**1.5;
        }
        const expected=sum*step*step*t*t,actual=morphologySolidAngle('ionized_gaussian',t,cut,m,profileValue);
        assert.ok(Math.abs(actual/expected-1)<.001,`${family}: ${actual/expected}`);
    }
});
test('position angle is stored in a fixed Galactic tangent frame, including both poles',()=>{
    for(const [lon,lat] of [[0,0],[359.999,30],[0,90],[257,-90]]){
        const p=prepareDeepSources([{id:'frame',kind:'emission_nebula',profile:'ionized_gaussian',profile_truncation:1,
            angular_scale_rad:.01,gal_lon:lon,gal_lat:lat,v_flux:.01,app_mag:5,morphology:shape('filament')}]);
        const s=p.sources[0],dot=s.axis.reduce((sum,v,i)=>sum+v*s.direction[i],0);
        assert.ok(Math.abs(dot)<1e-14);assert.ok(Math.abs(Math.hypot(...s.axis)-1)<1e-14);
        assert.ok(Math.abs(s.peak*s.solidAngle/s.flux-1)<1e-12);
    }
});
test('structured cloud support stays within the parent envelope without opaque masks',()=>{
    for(const family of MORPHOLOGY_FAMILIES){
        const m=shape(family);let lit=0;
        for(let i=0;i<300;i++){
            const a=i*2*Math.PI/300,x=Math.cos(a),y=Math.sin(a);
            assert.equal(morphologyWeight(x*1.01,y*1.01,'ionized_gaussian',1,m,profileValue),0);
            lit+=morphologyWeight(x*.4,y*.4,'ionized_gaussian',1,m,profileValue);
        }
        assert.ok(lit>0);
    }
});

test('local smoke placement changes the interior without imprinting texture on the outer rim',()=>{
    for(const family of ['turbulent','shell','blister']){
        const a={...shape(family),model:SOFT_MORPHOLOGY_MODEL},b={...a,phase_rad:a.phase_rad+1};let innerDifference=0;
        for(let i=0;i<48;i++){
            const phi=i*Math.PI/24,c=Math.cos(phi),s=Math.sin(phi)*a.axis_ratio;
            const value=(m,r)=>morphologyWeight(c*r,s*r,'ionized_gaussian',1,m,profileValue);
            assert.equal(value(a,.90),value(b,.90));
            innerDifference+=Math.abs(value(a,.35)-value(b,.35));
        }
        assert.ok(innerDifference>1e-5,`${family}: no local smoke structure`);
    }
});

test('filtered emission matches independent Python Gaussian convolution in fixed cloud coordinates',()=>{
    const refs=JSON.parse(readFileSync(new URL('./fixtures/nebula_filtered_reference.json',import.meta.url)));
    for(const r of refs.samples){
        const actual=morphologyWeight(r.x,r.y,r.profile,r.cut,r.morphology,profileValue);
        assert.ok(Math.abs(actual-r.value)<2e-7,`${r.morphology.family}: ${actual} vs ${r.value}`);
    }
});
test('mixed legacy and filtered catalogues assign independent atlas tiles without changing cluster flux',()=>{
    const make=(id,family,model)=>({id,kind:'emission_nebula',profile:'ionized_gaussian',profile_truncation:1,angular_scale_rad:.01,gal_lon:0,gal_lat:0,v_flux:.01,app_mag:5,morphology:{...shape(family),model}});
    const cluster={id:'cluster',kind:'open_cluster',profile:'plummer',angular_scale_rad:.01,gal_lon:0,gal_lat:0,v_flux:.1,app_mag:2.5};
    const objects=[make('old','shell',SOFT_MORPHOLOGY_MODEL),make('new','shell',MORPHOLOGY_MODEL),make('other','filament',MORPHOLOGY_MODEL),cluster],p=prepareDeepSources(objects);
    assert.equal(p.filterAtlas.count,2);assert.deepEqual(p.sources.map(s=>s.filterTile),[-1,0,1,-1]);
    assert.equal(p.sources[3].peak,prepareDeepSources([cluster]).sources[0].peak);
    const grid=filteredMorphologyGrid('ionized_gaussian',1,objects[1].morphology,profileValue);
    for(let y=0;y<grid.size;y++)for(let x=0;x<grid.size;x++)assert.equal(p.filterAtlas.values[y*p.filterAtlas.width+x],grid.values[y*grid.size+x]);
});
