import { createClient } from '@supabase/supabase-js';
const client=process.env.SUPABASE_URL&&process.env.SUPABASE_SERVICE_ROLE_KEY?createClient(process.env.SUPABASE_URL,process.env.SUPABASE_SERVICE_ROLE_KEY):null;
export const enabled=Boolean(client);
export async function savePrediction(row){if(!client)return null;const {data,error}=await client.from('prediction_history').insert(row).select().single();if(error)throw error;return data;}
export async function listPredictions(filters={}){if(!client)return {configured:false,rows:[]};let q=client.from('prediction_history').select('id,created_at,model_name,prediction,approval_probability,risk_level').order('created_at',{ascending:false}).limit(100);if(filters.prediction)q=q.eq('prediction',filters.prediction);if(filters.model)q=q.eq('model_name',filters.model);const {data,error}=await q;if(error)throw error;return {configured:true,rows:data};}
