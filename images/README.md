# Generations

Each GIF is its command's real output. Most use [a modern backbone](/README.md#modern-backbones), with `-freq 5` to keep every fifth step.

![](/images/alien_friend_by_Odilon_Redo_sd-turbo.gif?raw=true)

`uv run cgd -bb sd-turbo -gif "Alien friend by Odilon Redo"`

![](/images/Rolling_green_hills_Windows_XP_wallpaper.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo "Rolling green hills under a blue sky with white clouds, Windows XP wallpaper"`

![](/images/A_mushroom_in_a_mossy_forest_8K_HDR_photograph.gif?raw=true)

`uv run cgd -gif -freq 3 -size 256 -clip ViT-B/32+PE-Core-B-16 -n 90 -gl 0.99 "A mushroom in a mossy forest, 8K HDR photograph"`

![](/images/A_city_street_at_dusk_Kodak_Portra_400.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo "A city street at dusk, photo taken on Kodak Portra 400"`

![](/images/A_spaceship_over_a_desert_planet_1980s_movie_poster.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo "A spaceship over a desert planet, 1980s science fiction movie poster"`

![](/images/The_planet_Saturn_MIN_zoomed_out.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo --seed 1 "The planet Saturn|zoomed out:-0.3"`

Penalizing `zoomed out`. Seed 1 came in closest. Seed 0 drew the whole planet, over a caption strip of nonsense text.

![](/images/A_goose_running_across_a_field.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo "A goose running across a field"`

![](/images/A_golden_retriever_looking_at_the_camera.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo "A golden retriever looking at the camera"`

![](/images/A_glass_sphere_resting_on_a_mirror.gif?raw=true)

`uv run cgd -gif -freq 5 -bb sd-turbo "A glass sphere resting on a mirror"`
