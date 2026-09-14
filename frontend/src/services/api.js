const base=import.meta.env.VITE_API_URL||'http://localhost:3001/api';
export async function api(path,options={}){const res=await fetch(`${base}${path}`,{headers:{'Content-Type':'application/json'},...options});const body=await res.json();if(!res.ok)throw new Error(body.error||body.detail||'Request failed');return body.data;}
