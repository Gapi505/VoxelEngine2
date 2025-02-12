
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
struct PostProcessSettings {
    noise_strength: f32,
    frame: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
}
@group(0) @binding(2) var<uniform> settings: PostProcessSettings;

const SCANLINES: f32 = 200;

const HOR_RES: f32 = 420;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Chromatic aberration strength
    let offset_strength = settings.noise_strength;
    // Sample each color channel with an arbitrary shift
    let lowres_uv = vec2<f32>(round(in.uv.x*HOR_RES)/(HOR_RES), round(in.uv.y*SCANLINES)/SCANLINES);
    let dist = abs((lowres_uv.y - in.uv.y) * SCANLINES);
    let inv_dist = 1-dist;
    var out = vec4<f32>(
        textureSample(screen_texture, texture_sampler, lowres_uv).r,
        textureSample(screen_texture, texture_sampler, lowres_uv).g,
        textureSample(screen_texture, texture_sampler, lowres_uv).b,
        1.0
    );
    let frame = settings.frame;


    let uv_pix = vec2<f32>(lowres_uv.x*HOR_RES, lowres_uv.y*SCANLINES);
    let ind = uv_pix.y * SCANLINES + uv_pix.x + frame%123 * SCANLINES*HOR_RES;
    
    var noise_r = white_noise(ind*white_noise(ind));
    var noise_g = white_noise(ind + noise_r * 89123);
    var noise_b = white_noise(ind+ noise_g * 32490);
    
    var noise = vec4<f32>(noise_r, noise_g, noise_b, 1.);
    noise.r *= perlin_noise_1d(ind/332 + white_noise(uv_pix.y*SCANLINES + frame%121 * SCANLINES));
    noise.g *= perlin_noise_1d(ind/1324 + white_noise(uv_pix.y*SCANLINES + frame%324 * SCANLINES*10.43));
    noise.b *= perlin_noise_1d(ind/432 + white_noise(uv_pix.y*SCANLINES + frame%234 * SCANLINES*4.34));
    noise *= perlin_noise_1d(uv_pix.y/20 + white_noise(frame%2341)*30);
    noise *= perlin_noise_1d(uv_pix.x/600 + white_noise(frame%231)*35);
    noise.a = 1.;
    noise *= 2.;


    
    if (noise.r > (1-settings.noise_strength)){
        out.r = 0.;
    }
    if (noise.g > (1-settings.noise_strength)){
        out.g = 0.;
    }
    if (noise.b > (1-settings.noise_strength)){
        out.b = 0.;
    }

    out *= inv_dist;
    
    return out;
}

fn hash_u32(x: u32) -> f32 {
    var h: u32 = x;
    h = h * 747796405u + 2891336453u;
    h = (h >> 16u) ^ h;
    return f32(h & 0xFFFFu) / 65535.0; // Normalize to 0-1 range
}

fn white_noise(x: f32) -> f32 {
    return max(min(hash_u32(u32(x)), 1.),0.);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn gradient(h: f32, x: f32) -> f32 {
    return mix(-1.0, 1.0, h) * x;
}

fn perlin_noise_1d(x: f32) -> f32 {
    let xi: u32 = u32(floor(x));
    let xf: f32 = fract(x);

    // Add a tiny random offset to break deterministic patterns at whole numbers
    let offset: f32 = hash_u32(xi) * 0.1; // Offset max ±0.1 range
    let x_offset: f32 = xf + offset;

    let u: f32 = fade(x_offset);

    let g0: f32 = gradient(hash_u32(xi), x_offset);
    let g1: f32 = gradient(hash_u32(xi + 1u), x_offset - 1.0);

    return max(min(mix(g0, g1, u)/2. +0.5,1.),0.);
}
