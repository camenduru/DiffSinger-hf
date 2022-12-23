from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, \
    EnergyPredictor, FastspeechEncoder
from utils.cwt import cwt2f0
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0
from modules.fastspeech.fs2 import FastSpeech2


class FastspeechMIDIEncoder(FastspeechEncoder):
    def forward_embedding(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x


FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}


class FastSpeech2MIDI(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.midi_embed = Embedding(300, self.hidden_size, self.padding_idx)
        self.midi_dur_layer = Linear(1, self.hidden_size)
        self.is_slur_embed = Embedding(2, self.hidden_size)

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        ret = {}

        midi_embedding = self.midi_embed(kwargs['pitch_midi'])
        midi_dur_embedding, slur_embedding = 0, 0
        if kwargs.get('midi_dur') is not None:
            midi_dur_embedding = self.midi_dur_layer(kwargs['midi_dur'][:, :, None])  # [B, T, 1] -> [B, T, H]
        if kwargs.get('is_slur') is not None:
            slur_embedding = self.is_slur_embed(kwargs['is_slur'])
        encoder_out = self.encoder(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add ref style embed
        # Not implemented
        # variance encoder
        var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        elif hparams['use_spk_id']:
            spk_embed_id = spk_embed
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = self.spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = self.spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0

        # add dur
        dur_inp = (encoder_out + var_embed + spk_embed_dur) * src_nonpadding

        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])

        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp_origin = decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # add pitch and energy embed
        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        if hparams['use_pitch_embed']:
            pitch_inp_ph = (encoder_out + var_embed + spk_embed_f0) * src_nonpadding
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph)
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(pitch_inp, energy, ret)

        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding

        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret

    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        pitch_padding = mel2ph == 0
        if hparams['pitch_ar']:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp, f0 if self.training else None)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
            if hparams['use_uv'] and uv is None:
                uv = pitch_pred[:, :, 1] > 0

        # here f0_denorm for pitch prediction
        ret['f0_denorm'] = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)

        # here f0_denorm for mel prediction
        if self.training:
            mask = torch.full(uv.shape, hparams.get('mask_uv_prob', 0.)).to(f0.device)
            masked_uv = torch.bernoulli(mask).bool().to(f0.device)  # prob 的概率吐出一个随机uv.
            uv_masked = uv.bool() | masked_uv
            # print((uv.float()-uv_masked.float()).mean(dim=1))
            f0_denorm = denorm_f0(f0, uv_masked, hparams, pitch_padding=pitch_padding)
        else:
            f0_denorm = ret['f0_denorm']

        if pitch_padding is not None:
            f0[pitch_padding] = 0

        pitch = f0_to_coarse(f0_denorm)  # start from 0
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed


class FastSpeech2MIDIMasked(FastSpeech2MIDI):
    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        ret = {}

        midi_dur_embedding, slur_embedding = 0, 0
        if kwargs.get('midi_dur') is not None:
            midi_dur_embedding = self.midi_dur_layer(kwargs['midi_dur'][:, :, None])  # [B, T, 1] -> [B, T, H]
        if kwargs.get('is_slur') is not None:
            slur_embedding = self.is_slur_embed(kwargs['is_slur'])
        encoder_out = self.encoder(txt_tokens, 0, midi_dur_embedding, slur_embedding)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add ref style embed
        # Not implemented
        # variance encoder
        var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        elif hparams['use_spk_id']:
            spk_embed_id = spk_embed
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = self.spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = self.spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0

        # add dur
        dur_inp = (encoder_out + var_embed + spk_embed_dur) * src_nonpadding

        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])

        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        # expanded midi
        midi_embedding = self.midi_embed(kwargs['pitch_midi'])
        midi_embedding = F.pad(midi_embedding, [0, 0, 1, 0])
        midi_embedding = torch.gather(midi_embedding, 1, mel2ph_)
        print(midi_embedding.shape, decoder_inp.shape)
        midi_mask = torch.full(midi_embedding.shape, hparams.get('mask_uv_prob', 0.)).to(midi_embedding.device)
        midi_mask = 1 - torch.bernoulli(midi_mask).bool().to(midi_embedding.device)  # prob 的概率吐出一个随机uv.

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        decoder_inp += midi_embedding
        decoder_inp_origin = decoder_inp
        # add pitch and energy embed
        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        if hparams['use_pitch_embed']:
            pitch_inp_ph = (encoder_out + var_embed + spk_embed_f0) * src_nonpadding
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph)
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(pitch_inp, energy, ret)

        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding

        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret