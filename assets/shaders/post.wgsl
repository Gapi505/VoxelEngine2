
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
struct PostProcessSettings {
    noise_strength: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
}
@group(0) @binding(2) var<uniform> settings: PostProcessSettings;

const SCANLINES: f32 = 600;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Chromatic aberration strength
    let offset_strength = settings.noise_strength;
    // Sample each color channel with an arbitrary shift
    let lowers_uv = vec2<f32>(in.uv.x, floor(in.uv.y * SCANLINES)/SCANLINES);
    return vec4<f32>(
        textureSample(screen_texture, texture_sampler, lowers_uv).r,
        textureSample(screen_texture, texture_sampler, lowers_uv).g,
        textureSample(screen_texture, texture_sampler, lowers_uv).b,
        1.0
    );
}
