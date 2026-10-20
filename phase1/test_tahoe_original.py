"""Original 70M checkpoint diagnostic using a PyTorch SDPA adapter.
Synthetic binned inputs only; not proteomics training or author-backend equivalence.
Encoder/decoder operations follow the supplied upstream source.
Selected decoder classes below: Copyright (C) Tahoe Therapeutics 2025-2026.
"""
from __future__ import annotations
import json,math,hashlib,traceback,time,copy
from pathlib import Path
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
import yaml

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/tahoe_original/70m-model'
OUT=ROOT/'results/tahoe_original_gpu_test.json'
EXPECTED_SHA='ee1cdfa841066b9bce9c3705b112b17a2b459750d20696cbfedc6a3c397686c9'

def resolve_ffn_act_fn(config):
    return {'relu':nn.ReLU,'leaky_relu':nn.LeakyReLU,'sigmoid':nn.Sigmoid}[config['name']]()

class ContinuousValueEncoder(nn.Module):
    """Encode real number values to a vector using neural nets projection."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 512,
        activation: str = "relu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear2 = nn.Linear(d_model, d_model)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        if self.use_norm:
            x = self.norm(x)
        return self.dropout(x)

class ExprDecoder(nn.Module):
    """Consists of three linear functions and leaky-relu as an activation
    function."""

    def __init__(
        self,
        d_model: int,
        n_outputs: int = 1,
        n_layers: int = 2,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        d_in = d_model
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_in, d_model) for _ in range(n_layers)],
        )
        self.out_proj = nn.Linear(d_model, n_outputs)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """X is the output of the transformer, (batch, seq_len, d_model)"""
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        pred_value = self.out_proj(x)  # (batch, seq_len, n_outputs)
        if pred_value.shape[-1] == 1:
            pred_value = pred_value.squeeze(-1)
        return {"pred": pred_value}

class MVCDecoder(nn.Module):
    """Decoder for the masked value prediction for cell embeddings."""

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: str = "sigmoid",
        scaled_dot_product: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model
        self.scaled_dot_product = scaled_dot_product
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = resolve_ffn_act_fn({"name": query_activation})
            self.W = nn.Linear(d_model, d_in, bias=False)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(
                self.gene2query(gene_embs),
            )  # (batch, seq_len, embsize)
            inner_product_dimension = query_vecs.shape[-1]
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(
                2,
            )  # (batch, seq_len)
            if self.scaled_dot_product:
                pred_value = pred_value / torch.sqrt(
                    torch.tensor(inner_product_dimension, dtype=pred_value.dtype),
                )
            return {"pred": pred_value}
        else:
            raise ValueError(f"Unknown arch_style: {self.arch_style}")

class GeneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(62720,512,padding_idx=0)
        self.enc_norm=nn.LayerNorm(512)
    def forward(self,genes):
        return self.enc_norm(self.embedding(genes))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqkv=nn.Linear(512,1536)
        self.out_proj=nn.Linear(512,512)
    def forward(self,x,valid):
        b,l,_=x.shape
        q,k,v=self.Wqkv(x).reshape(b,l,3,8,64).permute(2,0,3,1,4).unbind(0)
        mask=None if bool(valid.all()) else valid[:,None,None,:]
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=False)
        return self.out_proj(a.transpose(1,2).reshape(b,l,512))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=Attention()
        self.up_proj=nn.Linear(512,2048);self.down_proj=nn.Linear(2048,512)
        self.norm1=nn.LayerNorm(512,eps=1e-5);self.norm2=nn.LayerNorm(512,eps=1e-5)
    def forward(self,x,valid):
        x=x+self.self_attn(self.norm1(x),valid)
        return x+self.down_proj(F.relu(self.up_proj(self.norm2(x))))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([Block() for _ in range(12)])
        self.norm=nn.LayerNorm(512,eps=1e-5)
    def forward(self,x,valid):
        for layer in self.layers:
            x=checkpoint(layer,x,valid,use_reentrant=False) if self.training else layer(x,valid)
        return self.norm(x)

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.gene_encoder=GeneEncoder();self.flag_encoder=nn.Embedding(2,512)
        self.expression_encoder=ContinuousValueEncoder(512,dropout=0.1,max_value=512,activation='relu',use_norm=True)
        self.transformer_encoder=Encoder()
        self.expression_decoder=ExprDecoder(512,n_outputs=1,n_layers=1,activation='leaky_relu')
        self.mvc_decoder=MVCDecoder(512,arch_style='inner product',query_activation='sigmoid',scaled_dot_product=True)
    def forward(self,genes,values,masked):
        tokens=self.gene_encoder(genes)
        encoded=self.expression_encoder(values).masked_fill(masked.unsqueeze(-1),0.0)
        flags=masked.unsqueeze(-1).to(tokens.dtype)*self.flag_encoder.weight[1].reshape(1,1,-1)
        hidden=self.transformer_encoder(tokens+encoded+flags,genes.ne(0))
        cell=hidden[:,0,:]
        return {'cell_emb':cell,'expr_preds':self.expression_decoder(hidden)['pred'],'mvc_output':self.mvc_decoder(cell,tokens)['pred']}

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.model=Core()
    def forward(self,genes,values,masked):
        return self.model(genes,values,masked)

def run():
    report={'scope':'Original checkpoint loaded into independently implemented SDPA adapter; synthetic binned input; no biological evaluation','backend_equivalence':'Not verified against upstream FlashAttention runtime','tests':[]}
    try:
        assert torch.cuda.is_available(),'CUDA unavailable'
        cfg=yaml.safe_load((MODEL/'model_config.yml').read_text())
        col=yaml.safe_load((MODEL/'collator_config.yml').read_text())
        for key,value in {'d_model':512,'n_heads':8,'n_layers':12,'vocab_size':62720,'use_glu':False,'norm_scheme':'pre','transformer_activation':'relu','cell_emb_style':'cls','expansion_ratio':4,'standard_scale_outputs':False}.items():
            assert cfg[key]==value,'Unsupported config: '+key
        assert not cfg['attn_config']['use_attn_mask'] and cfg['attn_config']['attn_pdrop']==0
        assert col['pad_token_id']==0 and col['num_bins']==51
        vocab=json.loads((MODEL/'vocab.json').read_text())
        assert '<cls>' in vocab and len(vocab)==62720
        h=hashlib.sha256()
        with (MODEL/'model.safetensors').open('rb') as f:
            for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
        assert h.hexdigest()==EXPECTED_SHA,'Unexpected checkpoint hash'
        state=load_file(str(MODEL/'model.safetensors'),device='cpu')
        model=Model()
        expected=model.state_dict()
        assert set(expected)==set(state),'Tensor-name mismatch: '+repr(set(expected)^set(state))
        assert all(expected[k].shape==state[k].shape for k in expected),'Tensor-shape mismatch'
        model.load_state_dict(state,strict=True)
        assert all(torch.equal(model.state_dict()[k],state[k]) for k in state),'Loaded values differ'
        report.update(checkpoint_sha256=h.hexdigest(),tensors=len(state),parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
        del state,expected
        model.cuda()
        print('Strict checkpoint load passed:',report['parameters'],'parameters',flush=True)
        torch.manual_seed(42)
        gene_ids=[int(v) for k,v in vocab.items() if not k.startswith('<')]
        for length in [256,1024]:
            model.eval()
            genes=torch.tensor([[vocab['<cls>']]+gene_ids[:length-1]],device='cuda')
            targets=torch.randint(1,51,(1,length),device='cuda').float();targets[:,0]=0
            masked=torch.rand(1,length,device='cuda')<0.5;masked[:,0]=False
            values=targets.masked_fill(masked,-1)
            # Verify efficient attention numerically against math on the same loaded model.
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                with sdpa_kernel(SDPBackend.MATH):reference=model(genes,values,masked)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):actual=model(genes,values,masked)
            discrepancies={k:float((actual[k].float()-reference[k].float()).abs().max()) for k in reference}
            for k in reference:
                torch.testing.assert_close(actual[k].float(),reference[k].float(),rtol=0.03,atol=0.03,msg='SDPA backend disagreement: '+k)
            del reference,actual
            model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
            scaler=torch.amp.GradScaler('cuda',init_scale=1.0)
            before=model.model.expression_decoder.out_proj.weight.detach().clone()
            successful=0;attempts=0;torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            while successful<2 and attempts<12:
                attempts+=1;optimizer.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),torch.autocast('cuda',dtype=torch.float16):
                    outputs=model(genes,values,masked)
                    loss=(F.mse_loss(outputs['expr_preds'][masked].float(),targets[masked])+F.mse_loss(outputs['mvc_output'][masked].float(),targets[masked]))/2
                assert torch.isfinite(loss),'Nonfinite loss'
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                assert all(p.grad is not None for p in model.parameters()),'Missing gradients'
                finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                scaler.step(optimizer);scaler.update()
                if finite:successful+=1
            assert successful==2,'No two finite-gradient updates after scaler retries'
            assert not torch.equal(before,model.model.expression_decoder.out_proj.weight),'No parameter update'
            assert len(optimizer.state)==len(list(model.parameters())),'Incomplete Adam state'
            torch.cuda.synchronize()
            row={'length':length,'status':'PASS','successful_updates':successful,'attempts':attempts,'loss':float(loss.detach()),'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'seconds':time.perf_counter()-start,'SDPA_max_abs_differences':discrepancies}
            report['tests'].append(row);print(json.dumps(row),flush=True)
            optimizer.zero_grad(set_to_none=True)
            del optimizer,outputs,loss,before,scaler
            torch.cuda.empty_cache()
        report['status']='PASS'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc) or repr(exc),traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2))
    print('Report:',OUT.resolve())
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':
    run()
