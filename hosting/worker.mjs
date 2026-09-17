/* global HTMLRewriter */

const worker = {
  async fetch(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", {
        status: 405,
        headers: { Allow: "GET, HEAD" },
      });
    }

    const response = await env.ASSETS.fetch(request);
    if (response.status !== 404) return withSocialMetadata(request, response);

    const url = new URL(request.url);
    const acceptsHtml = request.headers.get("Accept")?.includes("text/html") ?? false;
    const hasFileExtension = /\.[a-z0-9]+$/i.test(url.pathname);
    if (!acceptsHtml && hasFileExtension) return response;

    url.pathname = "/index.html";
    url.search = "";
    const fallback = await env.ASSETS.fetch(new Request(url, request));
    return withSocialMetadata(request, fallback);
  },
};

function withSocialMetadata(request, response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/html")) return response;

  const origin = new URL(request.url).origin;
  const title = "ActiveTrack — Your Reps. Counted.";
  const description = "Live and recorded-video basketball shot tracking with automatic hoop lock, ball tracking, and make/miss analysis.";
  const image = `${origin}/og.png`;

  return new HTMLRewriter()
    .on("head", {
      element(element) {
        element.append(
          `<meta name="description" content="${description}">` +
          `<meta property="og:type" content="website">` +
          `<meta property="og:title" content="${title}">` +
          `<meta property="og:description" content="${description}">` +
          `<meta property="og:image" content="${image}">` +
          `<meta name="twitter:card" content="summary_large_image">` +
          `<meta name="twitter:title" content="${title}">` +
          `<meta name="twitter:description" content="${description}">` +
          `<meta name="twitter:image" content="${image}">`,
          { html: true },
        );
      },
    })
    .transform(response);
}

export default worker;
