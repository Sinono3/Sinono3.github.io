import { RenderPlugin } from "@11ty/eleventy";
import syntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";

export default async function(eleventyConfig) {
		eleventyConfig.addPlugin(syntaxHighlight);
  	eleventyConfig.addPlugin(RenderPlugin);
  
  	eleventyConfig
    	.addPassthroughCopy("**/*.jpg")
    	.addPassthroughCopy("**/*.png")
    	.addPassthroughCopy("**/*.mp4")
    	.addPassthroughCopy("**/*.gif")
    	.addPassthroughCopy("**/*.svg")
    	.addPassthroughCopy({
  			"./public/": "/"
  		});
    eleventyConfig.addShortcode("video", function(path, caption, autoplay, extended) {
      autoplay = (autoplay !== undefined && autoplay) ? "autoplay" : "controls";
      extended = (extended !== undefined && extended) ? 'class="extended"' : "";
      caption = (caption !== undefined && caption != "") ? `<figcaption>
        ${caption}
      </figcaption>` : "";

      return `<figure ${extended}>
          <video ${autoplay} muted loop><source src=${path} type="video/mp4"></video>
          ${caption}
      </figure>`;
    });
    eleventyConfig.addShortcode("img", function(path, caption, extended) {
      extended = (extended !== undefined && extended) ? 'class="extended"' : "";
      caption = (caption !== undefined && caption != "") ? `<figcaption>
        ${caption}
      </figcaption>` : "";

      return `<figure ${extended}>
          <img src="${path}"/>
          ${caption}
      </figure>`;
    });
};

