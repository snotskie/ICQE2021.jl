using Glob
using Weave
cd(Base.source_dir())
weave("images_for_carl.jmd"; doctype="md2html", out_path=:pwd)
rm.(glob("jl_*"), recursive=true)