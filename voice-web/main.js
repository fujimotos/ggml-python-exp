import WaveSurfer from '/wavesurfer.js/dist/wavesurfer.esm.js'
import RegionsPlugin from '/wavesurfer.js/dist/plugins/regions.esm.js'
import TimelinePlugin from '/wavesurfer.js/dist/plugins/timeline.esm.js'

const COLOR_WAVE = 'rgb(100, 100, 100)';
const COLOR_WAVE_PROGRESS = 'rgb(200, 100, 100)';
const COLOR_REGION_ACTIVE = `rgba(196, 64, 0, 0.2)`;
const COLOR_REGION_INACTIVE = 'rgba(0, 0, 0, 0.1)';

/*
 * Save annotation data into LocalStorage
 */
function get_storage_key(ctx)
{
    return `${ctx.filename}/${ctx.select.value}`;
}

function save_to_storage(ctx)
{
    let regions = ctx.region_plugin.getRegions();
    let voice_segments = [];

    for (const region of regions) {
        if (region.end - region.start > 0) {
            voice_segments.push({
                start: region.start,
                end: region.end,
            });
        }
    }

    let key = get_storage_key(ctx);
    let blob = JSON.stringify({
        voice_segments: voice_segments
    });

    localStorage.setItem(key, blob);
    ctx.textarea.value = blob;
}

function load_from_storage(ctx)
{
    let key = get_storage_key(ctx);
    let blob = localStorage.getItem(key);

    if (blob) {
        ctx.region_plugin.clearRegions();

        let dict = JSON.parse(blob);
        for (const vs of dict.voice_segments) {
            ctx.region_plugin.addRegion({
                start: vs.start,
                end: vs.end,
            });
        }
        ctx.textarea.value = blob;
    } else {
        ctx.region_plugin.clearRegions();
        ctx.textarea.value = "";
    }
}

function open_new_audio(ctx, file)
{
    ctx.loading = true;
    ctx.wavesurfer.loadBlob(file).then(() => {
        ctx.filename = file.name;
        load_from_storage(ctx);
        ctx.loading = false;
    });
}

function set_active_region(ctx, new_region)
{
    if (ctx.active_region !== new_region) {
        unset_active_region(ctx);
        ctx.active_region = new_region;
        ctx.active_region.setOptions({ color: COLOR_REGION_ACTIVE });
    }
}

function unset_active_region(ctx)
{
    if (ctx.active_region) {
        ctx.active_region.setOptions({ color: COLOR_REGION_INACTIVE });
        ctx.active_region = null;
    }
}

/*
 * Main
 */
function main()
{
    const ctx = {
        wavesurfer: null,
        region_plugin: null,
        active_region: null,
        filename: null,
        textarea: null,
        fileinput: null,
        slider: null,
        select: null,
        loading: false,
    };

    ctx.region_plugin = RegionsPlugin.create();
    ctx.timeline_plugin = TimelinePlugin.create();

    ctx.wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: COLOR_WAVE,
        plugins: [ctx.region_plugin, ctx.timeline_plugin],
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
    });

    ctx.region_plugin.enableDragSelection({
        color: COLOR_REGION_INACTIVE,
    });

    ctx.region_plugin.on('region-created', (region) => {
        if (!ctx.loading) {
            set_active_region(ctx, region);
            save_to_storage(ctx);
        }
    });

    ctx.region_plugin.on('region-updated', (region) => {
        set_active_region(ctx, region);
        save_to_storage(ctx);
    });

    ctx.region_plugin.on('region-clicked', (region, e) => {
        e.stopPropagation();
        set_active_region(ctx, region);
    });

    ctx.region_plugin.on('region-double-clicked', (region, e) => {
        e.stopPropagation();
        if (region === ctx.active_region) {
            unset_active_region(ctx);
        }
        region.remove();
        save_to_storage(ctx);
    });

    ctx.wavesurfer.on('interaction', () => {
        unset_active_region(ctx);
    })

    ctx.wavesurfer.on('decode', () => {
        ctx.wavesurfer.zoom(ctx.slider.value);
    })

    /*
     * DOM Events
     */
    window.addEventListener('keydown', (e) => {
        if (e.key == ' ') {
            if (ctx.active_region) {
                ctx.active_region.play(true);
            } else if (ctx.wavesurfer.isPlaying()) {
                ctx.wavesurfer.pause();
            } else {
                ctx.wavesurfer.play();
            }
            e.preventDefault();
        }
    })

    window.addEventListener("DOMContentLoaded", () => {
        ctx.textarea = document.querySelector('#label');
        ctx.fileinput = document.querySelector('input[type="file"]');
        ctx.slider = document.querySelector('input[type="range"]');
        ctx.select = document.querySelector('select');

        ctx.slider.addEventListener("input", (e) => {
            ctx.wavesurfer.zoom(Number(e.target.value))
        });

        ctx.select.addEventListener("change", (e) => {
            load_from_storage(ctx)
        });

        ctx.fileinput.addEventListener("input", (e) => {
            e.target.blur();
            open_new_audio(ctx, ctx.fileinput.files[0]);
        });

        /*
         * This is for F5-key reload, Browsers can preserve
         * the state of the file input.
         */
        if (ctx.fileinput.files.length > 0) {
            open_new_audio(ctx, ctx.fileinput.files[0]);
        }
    });
};

main();
