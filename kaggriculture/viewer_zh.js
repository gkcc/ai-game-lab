(() => {
  'use strict';
  const names = {WHEAT:'小麦',CARROT:'胡萝卜',TOMATO:'番茄',STRAWBERRY:'草莓',MELON:'西瓜',
    GOOSE:'鹅',EGG:'鹅蛋',COW:'奶牛',MILK:'牛奶',SHEEP:'绵羊',WOOL:'羊毛',FERTILIZER:'肥料',
    BAKERY:'面包店',PIZZA_SHOP:'披萨店',BRUNCH_SPOT:'早午餐店',YARN_STORE:'毛线店',
    ICE_CREAM_SHOP:'冰淇淋店',PET_CAFE:'宠物咖啡馆',SMOOTHIE_SHOP:'果昔店',FARMERS_MARKET:'农贸市场'};
  const ops = {NORTH:'向上走',SOUTH:'向下走',EAST:'向右走',WEST:'向左走',PASS:'等待',
    PLANT:'播种',WATER:'浇水',HARVEST:'收获',FERTILIZE:'施肥',PICKUP:'从仓库取出',DROP:'卸货入仓',
    PLACE:'放置',FEED:'喂食',CARE:'照料动物',COLLECT_FERTILIZER:'收集肥料',DIG:'清理地块',
    BUILD_COOP:'建鹅舍',BUILD_PASTURE:'建牧场',BUY_SEED:'购买种子',BUY_ANIMAL:'购买动物',
    BUY_PRODUCT:'购买产品',SELL:'提交出售',HIRE:'雇一名工人',BUY_LAND:'购买土地'};
  const words = {Player:'玩家',Day:'天',Turn:'回合',Shed:'仓库',Market:'市场',Town:'城镇',
    'Town Center':'城镇中心','Welcome to Kaggriculture!':'欢迎来到农场对战！',
    Farmer:'主农夫',Farmhand:'工人',Weed:'杂草',Coop:'鹅舍',Pasture:'牧场',
    sprout:'幼苗',midgrowth:'生长中',ready:'可收获',fertilized:'已施肥',seed:'种子',
    'Loading...':'正在加载…','Waiting for replay data...':'等待回放数据…',
    Play:'播放',Pause:'暂停',Restart:'重新播放',Reset:'回到开头',Close:'关闭',
    'Step slider':'回合进度','Click to copy seed':'点击复制随机种子','Episode seed':'对局随机种子',
    'Click to copy Seed':'点击复制随机种子','Episode Seed':'对局随机种子',
    'Playback speed':'播放速度','Previous Step':'上一回合','Next Step':'下一回合',
    'Next step':'下一回合','Previous step':'上一回合',Speed:'速度',Seed:'随机种子',
    'Bakery':'面包店','Pizza Shop':'披萨店','Brunch Spot':'早午餐店','Yarn Store':'毛线店',
    'Ice Cream Shop':'冰淇淋店','Pet Cafe':'宠物咖啡馆','Smoothie Shop':'果昔店','Farmers Market':'农贸市场',
    "Farmer's Market":'农贸市场','Farmer’s Market':'农贸市场'};
  for (const [key,val] of Object.entries(names)) words[key[0]+key.slice(1).toLowerCase()]=val;
  const wordPattern = new RegExp(Object.keys(words).sort((a,b)=>b.length-a.length)
    .map(x=>x.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g');
  const translate = text => text.replace(wordPattern, x=>words[x]);
  const escape = text => String(text).replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
  const actionText = a => {
    const [op='PASS',item,n] = Array.isArray(a) ? a : ['PASS'];
    return (ops[op] || op) + (item ? '·'+(names[item] || item) : '') + (n !== undefined ? ' ×'+n : '');
  };
  const identity = i => i===0?'农夫':'工'+i;
  const positions = f => f ? [f.farmer,...(f.hands||[])] : [];
  const tileAt = (f,p) => p && f?.tiles?.[p[1]]?.[p[0]];
  const equal = (a,b) => JSON.stringify(a)===JSON.stringify(b);
  let lastStep=-1, lastShell=null, initialized=false;
  function localizeDom(root) {
    const walker = document.createTreeWalker(root,NodeFilter.SHOW_TEXT,{acceptNode(n){
      return n.parentElement?.closest('script,style,.zh-summary,.zh-bubbles,.zh-item-name,.zh-shop-name') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
    }});
    let node;
    while((node=walker.nextNode())) { const t=translate(node.nodeValue); if(t!==node.nodeValue)node.nodeValue=t; }
    root.querySelectorAll('[title],[aria-label],img[alt]').forEach(el=>{
      for(const attr of ['title','aria-label','alt']) {
        const v=el.getAttribute(attr); if(v) {const t=translate(v); if(v!==t)el.setAttribute(attr,t);}
      }
    });
    root.querySelectorAll('.market-item').forEach(el=>{
      const key=el.dataset.item;
      if(!el.querySelector('.zh-item-name')) {
        const label=document.createElement('span');label.className='zh-item-name';label.textContent=names[key]||key;el.prepend(label);
      }
    });
    root.querySelectorAll('.town-slot').forEach(el=>{
      const img=el.querySelector('img[title]'); if(!img)return;
      let label=el.querySelector('.zh-shop-name');
      if(!label){label=document.createElement('span');label.className='zh-shop-name';el.append(label);}
      label.textContent=translate(img.title || '');
    });
  }
  function resultText(op,before,after,pre,post,p,i,boundary) {
    if(op==='PASS')return '';
    const a=tileAt(pre,p), b=tileAt(post,p);
    if(['NORTH','SOUTH','EAST','WEST'].includes(op)) {
      if(boundary)return '（随后回仓）';
      return equal(positions(pre)[i],positions(post)[i])?'（未移动）':'';
    }
    // Day refresh resets water flags and deposits inventories; do not infer failure from these resets.
    if(boundary)return '（日终结算）';
    let changed;
    if(op==='WATER') changed=b?.kind==='PLANT'&&b.watered_today&&!a?.watered_today;
    if(op==='PLANT')changed=a===null&&b?.kind==='PLANT';
    if(op==='HARVEST') {
      const inv0=before?.private?.inventories?.[i]||{},inv1=after?.private?.inventories?.[i]||{};
      changed=Object.keys(inv1).some(k=>(inv1[k]||0)>(inv0[k]||0));
    }
    if(op==='DIG')changed=a!=null&&b===null;
    return changed===false?'（未生效）':'';
  }
  function separateLabels(shell) {
    for(const panel of shell.querySelectorAll('.farm-panel')) {
      const gridElement=panel.querySelector('.farm-grid');
      const grid=gridElement.getBoundingClientRect(), placed=[];
      const lines=document.createElementNS('http://www.w3.org/2000/svg','svg');
      lines.classList.add('zh-connectors');lines.setAttribute('viewBox',`0 0 ${grid.width} ${grid.height}`);
      gridElement.append(lines);
      for(const label of panel.querySelectorAll('.zh-bubble')) {
        const box=label.getBoundingClientRect();let choice=null;
        for(const dy of [0,-18,18,-36,36,-54,54,-72,72,-90,90]) {
          const x=Math.max(grid.left,Math.min(box.left,grid.right-box.width));
          const y=Math.max(grid.top,Math.min(box.top+dy,grid.bottom-box.height));
          const candidate={left:x,top:y,right:x+box.width,bottom:y+box.height};
          if(!placed.some(p=>candidate.left<p.right+2&&candidate.right>p.left-2&&candidate.top<p.bottom+2&&candidate.bottom>p.top-2)) {
            choice=candidate;break;
          }
        }
        if(choice){
          label.style.transform=`translate(${choice.left-box.left}px,${choice.top-box.top}px)`;placed.push(choice);
          if(Math.abs(choice.left-box.left)>2||Math.abs(choice.top-box.top)>2){
            const cell=label.closest('.cell').getBoundingClientRect();
            const line=document.createElementNS('http://www.w3.org/2000/svg','line');
            line.setAttribute('x1',cell.left+cell.width/2-grid.left);line.setAttribute('y1',cell.top+cell.height/2-grid.top);
            line.setAttribute('x2',choice.left+box.width/2-grid.left);line.setAttribute('y2',choice.top+box.height/2-grid.top);
            lines.append(line);
          }
        }
        else placed.push(box);
      }
    }
  }
  function update() {
    const replay=window.kaggle?.environment, shell=document.querySelector('.kaggriculture-container');
    if(!replay?.steps?.length||!shell)return;
    const range=document.querySelector('input[type=range]');
    if(!range)return;
    if(!initialized){
      // The bundled player starts automatically even when the HTML requests playing=false.
      // Start paused so the user can read the annotations; normal playback stays available.
      if(window.kaggle.playing===false)document.querySelector('button[aria-label="Pause"]')?.click();
      initialized=true;
    }
    const step=Number(range.value);
    if(!Number.isInteger(step)||step<0||step>=replay.steps.length)return;
    if(step===lastStep&&shell===lastShell)return;
    lastStep=step;lastShell=shell;
    localizeDom(document.getElementById('app'));
    const current=replay.steps[step], previous=replay.steps[Math.max(0,step-1)];
    const obs=current[0].observation, prevObs=previous[0].observation;
    const boundary=step>0&&obs.day!==prevObs.day;
    const end=step===replay.steps.length-1;
    shell.querySelectorAll('.zh-bubbles,.zh-connectors').forEach(el=>el.remove());
    let summary=shell.querySelector('.zh-summary');
    if(!summary){summary=document.createElement('section');summary.className='zh-summary';shell.append(summary);}
    const header=step===0?'开局：准备开始':`刚执行：第 ${prevObs.day+1} 天，第 ${prevObs.hour+1} 回合`;
    let html=`<h2>${header}</h2><p class="zh-help">气泡与下方编号对应：农夫＝主农夫，工1、工2…＝当天雇工。显示刚执行的动作；“未生效”表示状态未发生相应变化。市场栏显示订单，不代表全部成交。</p>`;
    if(boundary)html+='<p class="zh-status">一天结束：携带物自动入仓（超容量丢失），农夫回仓，雇工下班。田中保留紫色标记，说明下班前的动作。</p>';
    if(end)html+='<p class="zh-status">本季结束：比较双方现金决定胜负。最终状态保留最后一回合的动作说明。</p>';
    html+='<div class="zh-columns">';
    for(let team=0;team<2;team++) {
      const after=current[team]?.observation, before=previous[team]?.observation;
      const farm=obs.farms[team], old=prevObs.farms[team];
      const a=current[team]?.action||{}, units=positions(farm), oldUnits=positions(old);
      const commands=[a.farmer||['PASS'],...(a.hands||[])];
      const rows=[], badges=new Map();
      const add=(p,label,op)=>{
        if(!p)return; const key=p.join(','); if(!badges.has(key))badges.set(key,[]);
        badges.get(key).push({label,op});
      };
      const count=step===0?units.length:Math.max(units.length,oldUnits.length);
      for(let i=0;i<count;i++) {
        const command=commands[i]||['PASS'],op=command[0]||'PASS';
        let text,position=units[i],tag=op;
        if(step===0)text='准备开始';
        else if(boundary){text=(i===0?'回仓；此前':'下班；此前')+actionText(command);position=i===0?units[0]:oldUnits[i];tag='RESET';}
        else if(i>=oldUnits.length){text='新雇工到岗';tag='RESET';}
        else {
          text=actionText(command)+resultText(op,before,after,old,farm,oldUnits[i],i,boundary);
          if(text.includes('未生效')||text.includes('未移动'))tag='FAILED';
        }
        const coord=position?`（${position[0]+1}列，${position[1]+1}行）`:'';
        rows.push(`<span>${identity(i)}：${escape(text)} ${coord}</span>`);
        add(position,identity(i)+'·'+text,tag);
      }
      const panel=shell.querySelector(`.farm-panel[data-player="${team+1}"]`);
      for(const [coord,items] of badges) {
        const [x,y]=coord.split(',');
        const cell=panel?.querySelector(`.cell[data-col="${x}"][data-row="${y}"]`);
        if(!cell)continue;
        const box=document.createElement('div');box.className='zh-bubbles';
        // Keep every co-located unit visible rather than overwriting it as the base renderer does.
        box.innerHTML=items.map(v=>`<span class="zh-bubble" data-op="${escape(v.op)}">${escape(v.label)}</span>`).join('');
        cell.append(box);
      }
      const orders=(step===0?[]:a.market||[]).map(actionText);
      html+=`<div class="zh-team"><h3>玩家 ${team+1} · 现金 ${Math.floor(farm.money)} · 当前雇工 ${farm.hands.length} 人</h3><div class="zh-roster">${rows.join('')}</div><div class="zh-market">本回合市场订单：${orders.length?orders.map(escape).join('；'):'无'}</div></div>`;
    }
    html+='</div><div class="zh-buttons"><button data-zh-jump="first">回到开局</button><button data-zh-jump="prev">上一回合</button><button data-zh-jump="next">下一回合</button><button data-zh-jump="day">下一天</button></div>';
    summary.innerHTML=html;
    separateLabels(shell);
    summary.querySelectorAll('[data-zh-jump]').forEach(b=>b.addEventListener('click',()=>{
      const tpd=replay.configuration?.turnsPerDay||24;
      const target={first:0,prev:step-1,next:step+1,day:(Math.floor(step/tpd)+1)*tpd}[b.dataset.zhJump];
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(
        range,String(Math.max(0,Math.min(replay.steps.length-1,target))));
      range.dispatchEvent(new Event('input',{bubbles:true}));
      range.dispatchEvent(new Event('change',{bubbles:true}));
    }));
  }
  function loop(){update();requestAnimationFrame(loop);}
  window.addEventListener('resize',()=>{lastStep=-1;});
  requestAnimationFrame(loop);
})();
