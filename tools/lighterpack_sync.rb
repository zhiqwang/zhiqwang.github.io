#!/usr/bin/env ruby
# frozen_string_literal: true

require "cgi"
require "fileutils"
require "net/http"
require "time"
require "uri"
require "yaml"

class LighterpackSync
  DEFAULT_OUTPUT_DIR = File.expand_path("../_data/lighterpack", __dir__)

  def initialize(pack_ref, output_path = nil)
    @pack_id = normalize_pack_id(pack_ref)
    @source_url = URI("https://lighterpack.com/r/#{@pack_id}")
    @output_path = output_path || File.join(DEFAULT_OUTPUT_DIR, "#{@pack_id}.yml")
  end

  def run
    html = fetch_html
    pack = parse_pack(html)
    FileUtils.mkdir_p(File.dirname(@output_path))
    File.open(@output_path, "w:UTF-8") do |file|
      file.write(YAML.dump(pack))
    end
    puts "Wrote #{@output_path}"
  end

  private

  def normalize_pack_id(pack_ref)
    input = pack_ref.to_s.strip
    abort("Missing LighterPack pack id or URL") if input.empty?

    return input if input.match?(/\A[a-zA-Z0-9]+\z/)

    uri = URI(input)

    unless uri.host&.match?(/\Alighterpack\.com\z/i)
      abort("Unsupported LighterPack URL: #{input}")
    end

    match = uri.path.match(%r{\A/r/([a-zA-Z0-9]+)\z})
    abort("Could not extract pack id from #{input}") unless match

    match[1]
  rescue URI::InvalidURIError
    abort("Unsupported LighterPack reference: #{input}")
  end

  def fetch_html
    response = http_get(@source_url)

    return normalize_utf8(response.body) if response.is_a?(Net::HTTPSuccess)

    abort("Failed to fetch #{@source_url}: #{response.code} #{response.message}")
  end

  def http_get(uri, limit = 5)
    abort("Too many redirects while fetching #{uri}") if limit.zero?

    response = Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", open_timeout: 10, read_timeout: 20) do |http|
      request = Net::HTTP::Get.new(uri)
      request["User-Agent"] = "zhiqwang.github.io lighterpack sync"
      http.request(request)
    end

    return response unless response.is_a?(Net::HTTPRedirection)

    location = URI(response["location"])
    location = uri + response["location"] unless location.absolute?
    http_get(location, limit - 1)
  end

  def parse_pack(html)
    summary = parse_summary_categories(html)
    categories = parse_category_blocks(html, summary)
    total = parse_total(html, categories)

    {
      "id" => @pack_id,
      "title" => extract_required(html, %r{<h1 class="lpListName">(.*?)</h1>}m, "list title"),
      "source_url" => @source_url.to_s,
      "snapshot_at" => Time.now.utc.iso8601,
      "total" => total,
      "chart_css" => build_chart_css(categories, total["grams"]),
      "categories" => categories
    }
  end

  def parse_summary_categories(html)
    categories = {}

    html.scan(%r{<li class="lpTotalCategory lpRow" id="total_(\d+)" category="(\d+)">(.*?)</li>}m) do |_total_id, category_id, body|
      name = extract_required(body, %r{<span class="lpCell">\s*(.*?)\s*</span>}m, "category name")
      color = extract_required(body, /background-color:\s*([^";]+)/, "category color")
      value = extract_required(body, %r{<span class="lpDisplaySubtotal"[^>]*>(.*?)</span>}m, "category weight")
      unit = extract_required(body, %r{<span class="lpSubtotalUnit">(.*?)</span>}m, "category unit")
      mg = extract_required(body, /lpDisplaySubtotal"\s+mg="(\d+)"/, "category mg").to_i

      categories[category_id] = {
        "id" => category_id,
        "name" => name,
        "color" => color,
        "weight" => weight_hash(mg, value, unit),
        "item_count" => 0,
        "items" => []
      }
    end

    abort("No LighterPack categories found for #{@pack_id}") if categories.empty?

    categories
  end

  def parse_category_blocks(html, summary)
    matches = html.enum_for(:scan, /<li class="lpCategory" id="(\d+)">/).map do
      [Regexp.last_match.begin(0), Regexp.last_match[1]]
    end

    abort("No category blocks found for #{@pack_id}") if matches.empty?

    matches.each_with_index.map do |(start_index, category_id), index|
      end_index = index + 1 < matches.length ? matches[index + 1][0] : html.length
      block = html[start_index...end_index]
      category = summary.fetch(category_id) do
        abort("Missing summary row for category #{category_id} in #{@pack_id}")
      end

      item_starts = block.enum_for(:scan, /<li class="lpItem\b/).map { Regexp.last_match.begin(0) }
      footer_index = block.index('<li class="lpFooter"') || block.length

      items = item_starts.each_with_index.map do |item_start, item_index|
        item_end = item_index + 1 < item_starts.length ? item_starts[item_index + 1] : footer_index
        parse_item(block[item_start...item_end])
      end

      category["items"] = items
      category["item_count"] = items.sum { |item| item["quantity"] }
      category
    end
  end

  def parse_item(block)
    name = extract_required(block, %r{<span class="lpName">\s*(.*?)\s*</span>}m, "item name")
    description = extract_optional(block, %r{<span class="lpDescription">\s*(.*?)\s*</span>}m)
    value = extract_required(block, %r{<span class="lpWeight">\s*(.*?)\s*</span>}m, "item weight")
    unit = extract_required(block, %r{<span class="lpDisplay">\s*(.*?)\s*</span>}m, "item unit")
    mg = extract_required(block, /<input type="hidden" class="lpMG" value="(\d+)"/, "item mg").to_i
    quantity = extract_required(block, %r{<span class="lpQtyCell lpNumber"[^>]*>\s*(.*?)\s*</span>}m, "item quantity").to_i

    {
      "name" => name,
      "description" => description,
      "weight" => weight_hash(mg, value, unit),
      "quantity" => quantity
    }
  end

  def parse_total(html, categories)
    match = html.match(%r{<li class="lpRow lpFooter lpTotal">.*?<span class="lpTotalValue" title="(\d+) items">(.*?)</span>.*?<input type="hidden" class="lpMG" value="(\d+)"/>.*?<span class="lpDisplay">(.*?)</span>}m)
    abort("Missing total summary for #{@pack_id}") unless match

    item_count = match[1].to_i
    value = clean_text(match[2])
    mg = match[3].to_i
    unit = clean_text(match[4])

    {
      "item_count" => item_count.zero? ? categories.sum { |category| category["item_count"] } : item_count,
      "mg" => mg,
      "grams" => mg / 1000.0,
      "value" => value,
      "unit" => unit,
      "display" => "#{value} #{unit}"
    }
  end

  def build_chart_css(categories, total_grams)
    return "#cbd5e1 0% 100%" if total_grams.to_f <= 0

    start_percent = 0.0

    categories.map.with_index do |category, index|
      weight = category.dig("weight", "grams").to_f
      finish_percent = if index == categories.length - 1
                         100.0
                       else
                         start_percent + ((weight / total_grams) * 100.0)
                       end

      segment = "#{category["color"]} #{format_percent(start_percent)}% #{format_percent(finish_percent)}%"
      start_percent = finish_percent
      segment
    end.join(", ")
  end

  def weight_hash(mg, value, unit)
    clean_value = clean_text(value)
    clean_unit = clean_text(unit)

    {
      "mg" => mg,
      "grams" => mg / 1000.0,
      "value" => clean_value,
      "unit" => clean_unit,
      "display" => "#{clean_value} #{clean_unit}"
    }
  end

  def extract_required(text, pattern, label)
    match = text.match(pattern)
    abort("Missing #{label} while parsing #{@pack_id}") unless match

    clean_text(match[1])
  end

  def extract_optional(text, pattern)
    match = text.match(pattern)
    return "" unless match

    clean_text(match[1])
  end

  def clean_text(text)
    normalize_utf8(CGI.unescapeHTML(text.to_s.gsub(/<[^>]+>/, " ").gsub(/\s+/, " ").strip))
  end

  def normalize_utf8(text)
    text.to_s.force_encoding("UTF-8").encode("UTF-8", invalid: :replace, undef: :replace, replace: "")
  end

  def format_percent(number)
    format("%.2f", number).sub(/\.0+\z/, "").sub(/(\.\d*[1-9])0+\z/, '\\1')
  end
end

if ARGV.empty?
  abort("Usage: ruby tools/lighterpack_sync.rb PACK_ID_OR_URL [OUTPUT_PATH]")
end

LighterpackSync.new(ARGV[0], ARGV[1]).run