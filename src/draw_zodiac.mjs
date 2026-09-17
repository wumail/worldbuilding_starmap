import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {catalogueStars,SOURCE} from './build_zodiac_candidates.mjs';
import {generateDraw,prepareDraw,DEFAULT_SEED,DEFAULT_STYLE} from '../web/constellations/generator.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),options={seed:DEFAULT_SEED,style:DEFAULT_STYLE};
for(let i=0;i<args.length;i++){
    if(args[i]==='--sampling-half-width'&&args[i+1]){options.samplingHalfWidthDegrees=Number(args[++i]);continue;}
    if(!['--seed','--style','--out'].includes(args[i])||!args[i+1])throw Error('用法：node src/draw_zodiac.mjs --seed terrax-001 --style rich --sampling-half-width 40 --out output/zodiac_draws/terrax-001.json');
    options[args[i].slice(2)]=args[++i];
}
if(!options.out)throw Error('请通过 --out 指定新的候选文件路径；不会覆盖初稿或其他抽卡结果。');
const target=path.resolve(options.out),bytes=fs.readFileSync(path.join(root,SOURCE)),sha256=crypto.createHash('sha256').update(bytes).digest('hex');
await prepareDraw(options);
const data=generateDraw(catalogueStars(JSON.parse(bytes)),{catalogue:SOURCE,sha256},options);
const record={id:`cli-${crypto.createHash('sha256').update(JSON.stringify(data.recipe)).digest('hex').slice(0,12)}`,title:`种子 ${data.recipe.seed}`,recipe:data.recipe,locks:[],notes:{},favourite:false};
fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,JSON.stringify({format:'terrax-constellation-round',formatVersion:1,record,data},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({target,seed:data.recipe.seed,members:data.selectedExtendedCount,regions:data.regions.length,sourceHash:sha256}));
