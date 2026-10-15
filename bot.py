import discord
from discord.ext import commands
import os
import json
import re
import asyncio
import time
import unicodedata
import base64
from datetime import timedelta
from dotenv import load_dotenv
from groq import Groq

# .env betöltése: pentagon/.env (a jarvis mappa szülője), majd a jelenlegi mappa
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Képek olvasásához látó modell kell (a .env-ben felülírható: VISION_MODEL=...)
# A Groq percenkénti token-limitje a modelledre (console.groq.com/settings/limits). Ha nagyobb a limited, írd át .env-ben.
TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))
EFFECTIVE_TPM = int(TPM_LIMIT * 0.85)   # tartalék a becslés pontatlansága miatt
CHARS_PER_TOKEN = 2.6                   # óvatos becslés magyar szöveghez és emojikhoz
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
VISION_FALLBACK = "qwen/qwen3.6-27b"  # ha az első modell nem érhető el

if not DISCORD_TOKEN:
    raise ValueError("Hiányzik a DISCORD_BOT_TOKEN a .env fájlból!")
if not GROQ_API_KEY:
    raise ValueError("Hiányzik a GROQ_API_KEY a .env fájlból!")

# Opcionális: .env-ben ALLOWED_USER_IDS=123,456 -> ők akkor is használhatják, ha nem adminok
ALLOWED_USER_IDS = {
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit()
}

# Opcionális: AI_CHANNEL_IDS=123,456 -> sima üzenetre csak ezekben a csatornákban reagál
# (ha üres, akkor minden csatornában, de csak adminoknak)
AI_CHANNEL_IDS = {
    int(x) for x in os.getenv("AI_CHANNEL_IDS", "").split(",") if x.strip().isdigit()
}

intents = discord.Intents.default()
intents.guilds = True
intents.members = True          # Privileged – Developer Portal-on be kell kapcsolni
intents.message_content = True  # Privileged – Developer Portal-on be kell kapcsolni

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
# Több Groq kulcs támogatása (GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3 ...): ha az egyik eléri a limitet, vált a következőre
GROQ_KEYS = [GROQ_API_KEY] + [
    v for k, v in sorted(os.environ.items())
    if re.fullmatch(r"GROQ_API_KEY_\d+", k) and v.strip()
]
GROQ_CLIENTS = [Groq(api_key=k.strip()) for k in GROQ_KEYS]
_key_state = {"idx": 0}

# csatornánkénti rövid beszélgetés-memória
HISTORY: dict[int, list[dict]] = {}
HISTORY_LIMIT = 8


# ============================================================
# 1. SEGÉDFÜGGVÉNYEK
# ============================================================

def strip_accents(text: str) -> str:
    """Kisbetű, ékezetek nélkül (pl. 'töröld' és 'torold' ugyanaz)."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c)).lower()


def resolve_mentions(guild: discord.Guild, text: str) -> str:
    """A Discord említéseket (<@&id>, <#id>, <@id>) névvé alakítja, hogy az AI értse."""
    def role(m):
        r = guild.get_role(int(m.group(1)))
        return r.name if r else m.group(0)

    def chan(m):
        c = guild.get_channel(int(m.group(1)))
        return c.name if c else m.group(0)

    def mem(m):
        u = guild.get_member(int(m.group(1)))
        return u.display_name if u else m.group(0)

    text = re.sub(r"<@&(\d+)>", role, text)
    text = re.sub(r"<#(\d+)>", chan, text)
    text = re.sub(r"<@!?(\d+)>", mem, text)
    return text


def norm(s: str) -> str:
    """Kisbetű, emoji/szóköz/kötőjel nélkül – a laza névegyeztetéshez."""
    return re.sub(r"[\W_]+", "", str(s).lower())


def find_by_name(items, name, attr="name"):
    """Név alapú keresés: pontos -> kis/nagybetűfüggetlen -> emoji nélküli -> részleges."""
    if name is None:
        return None
    name = str(name)
    items = list(items)
    for it in items:
        if getattr(it, attr) == name:
            return it
    for it in items:
        if getattr(it, attr).lower() == name.lower():
            return it
    n = norm(name)
    if not n:
        return None
    for it in items:
        if norm(getattr(it, attr)) == n:
            return it
    for it in items:
        if n in norm(getattr(it, attr)):
            return it
    return None


def find_channel(guild, name, kinds=None):
    chans = [c for c in guild.channels if not isinstance(c, discord.CategoryChannel)]
    if kinds:
        chans = [c for c in chans if isinstance(c, kinds)]
    if name is not None and str(name).strip("<#>").isdigit():
        c = guild.get_channel(int(str(name).strip("<#>")))
        if c:
            return c
    return find_by_name(chans, name)


def find_category(guild, name):
    return find_by_name(guild.categories, name)


def find_role(guild, name):
    if name is None:
        return None
    if str(name).lower().strip() in ("@everyone", "everyone"):
        return guild.default_role
    if str(name).strip("<@&>").isdigit():
        r = guild.get_role(int(str(name).strip("<@&>")))
        if r:
            return r
    return find_by_name([r for r in guild.roles], name)


def find_member(guild, name):
    if name is None:
        return None
    s = str(name)
    if s.strip("<@!>").isdigit():
        m = guild.get_member(int(s.strip("<@!>")))
        if m:
            return m
    m = find_by_name(guild.members, s, "display_name")
    if m:
        return m
    return find_by_name(guild.members, s, "name")


def parse_color(value) -> discord.Colour:
    try:
        return discord.Colour(int(str(value).replace("#", ""), 16))
    except Exception:
        return discord.Colour.default()


def make_permissions(names) -> discord.Permissions:
    perms = discord.Permissions.none()
    for n in names or []:
        if n in discord.Permissions.VALID_FLAGS:
            setattr(perms, n, True)
    return perms


def split_message(text: str, limit: int = 1900):
    text = str(text)
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


def get_guild_state(guild: discord.Guild, level: int = 0) -> dict:
    """Kompakt szerverállapot az AI-nak. level 0: tagokkal, 1: tagok nélkül, 2: csak a legfontosabb."""
    def label(ch):
        return ("🔊" + ch.name) if isinstance(ch, (discord.VoiceChannel, discord.StageChannel)) else ch.name

    state = {
        "szerver": guild.name,
        "jelmagyarázat": "🔊 előtag = hangcsatorna (nem része a névnek)",
        "kategóriák": {cat.name: [label(c) for c in cat.channels] for cat in guild.categories},
        "kategória_nélküli_csatornák": [
            label(c) for c in guild.channels
            if c.category is None and not isinstance(c, discord.CategoryChannel)
        ],
        "rangok_fentről_lefelé": [
            r.name for r in sorted(guild.roles, key=lambda x: -x.position)
            if not r.is_default() and not r.managed
        ],
    }
    if level == 0:
        state["tagok"] = [
            f"{m.display_name} | " + ", ".join(r.name for r in m.roles if not r.is_default())
            for m in guild.members[:30]
        ]
    if level >= 2:
        state["kategóriák"] = list(state["kategóriák"].keys())
    return state


# ============================================================
# 2. AI – Groq
# ============================================================

SYSTEM_PROMPT = """Te a Pentagon AI vagy, egy mindentudó Discord szerverkezelő és beszélgető AI.
A felhasználó természetes nyelven (általában magyarul) beszél hozzád. Ha csak beszélget, válaszolj kedvesen, természetesen.
Ha szerverműveletet kér, add vissza a végrehajtandó műveleteket.

A válaszod MINDIG kizárólag érvényes JSON objektum legyen:
{
  "reply": "Rövid, természetes válasz a felhasználónak (a felhasználó nyelvén)",
  "actions": [ ... ]
}
Csak beszélgetésnél az "actions" legyen üres lista.

ELÉRHETŐ MŰVELETEK (az "action" mező értéke és paraméterei):

CSATORNÁK / KATEGÓRIÁK
- create_channel: name, channel_type ("text"|"voice"|"stage"|"forum"|"announcement"), category (név vagy null), topic (opcionális), private (bool, opcionális), allowed_roles (lista rangnevekkel, ha private)
- delete_channel: name
- rename_channel: old_name, new_name
- set_topic: channel_name, topic
- move_channel: channel_name, category (név vagy null)
- set_channel_position: channel_name, position
- set_slowmode: channel_name, seconds
- lock_channel: channel_name (a @everyone nem írhat)
- unlock_channel: channel_name
- hide_channel: channel_name (a @everyone nem látja)
- show_channel: channel_name
- set_channel_permission: channel_name, role_name ("@everyone" vagy rang), permission (discord jogosultság neve, pl. send_messages, view_channel, connect, speak, add_reactions, attach_files), allow (true/false/null=alaphelyzet)
- create_category: name, private (bool), allowed_roles (lista)
- delete_category: name, delete_channels (bool: a benne lévő csatornákat is törölje)
- rename_category: old_name, new_name
- clone_channel: name, new_name

ÜZENETEK
- send_message: channel_name (null = az aktuális csatorna), text, embed (bool), title (embed cím), color (hex)
- purge_messages: channel_name (null = aktuális), amount (1-500)
- create_invite: channel_name, max_age_seconds (0=soha), max_uses (0=korlátlan)

RANGOK
- create_role: name, color (hex pl. "FF0000"), hoist (bool), mentionable (bool), permissions (lista jogosultságnevekkel pl. ["manage_messages","kick_members"]), administrator (bool), all_permissions (bool: MINDEN jogosultság, az összes)
- delete_role: name
- edit_role: name, new_name, color, hoist, mentionable, add_permissions (lista), remove_permissions (lista), all_permissions (bool: MINDEN jogosultságot megad), clear_permissions (bool: minden jogosultságot elvesz)
- add_role: member_name, role_name
- remove_role: member_name, role_name

TAGOK
- ban_member: member_name, reason
- unban_member: user (felhasználónév vagy ID), reason
- kick_member: member_name, reason
- timeout_member: member_name, minutes, reason
- untimeout_member: member_name
- set_nickname: member_name, nickname (null = törlés)
- move_member_voice: member_name, channel_name (null = kidobja a hangcsatornából)

SZERVER
- rename_guild: new_name

SZABÁLYOK:
- Használd PONTOSAN a szerver-állapotban látható neveket meglévő elemek hivatkozásához.
- Új csatorna/rang/kategória nevében használj emojit, ha a felhasználó kéri vagy ha jól áll (pl. "💬│csevego"). A text csatornanevek kisbetűsek, kötőjellel lesznek; ez rendben van.
- A műveletek sorrendje számít: előbb a kategóriát/rangot hozd létre, utána a csatornákat, amik használják.
- Összetett kéréseknél (pl. "csinálj egy gaming szervert") generálj sok műveletet: kategóriák, csatornák, rangok, jogosultságok.
- Ha valami nem egyértelmű, kérdezz vissza a "reply"-ban és legyen az actions üres.
- Ha a kért dolog nem megoldható a fenti műveletekkel, mondd el őszintén a reply-ban.
- Ne találj ki nem létező műveletet.
- Ha a felhasználó "minden jogot / összes permissiont / mindent" kér egy rangnak, használd az edit_role (vagy create_role) all_permissions: true mezőjét. NE sorolj fel jogokat kézzel és ne csak administrator-t adj. Ha minden jogot el kell venni: clear_permissions: true.
- Ha a felhasználó csak egy hosszú számot (ID-t) ad meg egy korábbi kérésre válaszul, használd azt a megfelelő name/…_name mezőben változatlanul (a rendszer feloldja), és hajtsd végre a korábbi kérést. Ne kérdezz vissza az ID miatt.
- Ha a felhasználó ékezet nélkül ír vagy elgépel egy nevet, használd a szerver-állapotban látható legközelebbi nevet.
- SOHA ne írj kódot, programot, projektet, útmutatót, telepítési leírást vagy markdown-t. Te magad hajtod végre a kért dolgokat az actions listával.
- A "reply" szerverműveleteknél legyen MAX 2-3 rövid mondat, sima szöveg.
- Te általános segítő is vagy, nem csak szerverkezelő. Ha házi feladatot, kérdést, fordítást, számolást vagy magyarázatot kérnek (akár képről vagy fájlból), SOHA ne mondd, hogy ez nem szerverkezelés: olvasd le a feladatot, oldd meg, és a megoldást írd a "reply" mezőbe. Ilyenkor az actions üres.
- Képen lévő feladatoknál sorszám szerint haladj, adj pontos válaszokat (az angol nyelvi feladatoknál a válasz angolul is lehet, a magyarázat magyarul). Ha valami nem olvasható a képen, azt mondd meg őszintén.
- Ilyen tartalmi válasznál a reply lehet hosszabb (max kb. 1500 szó), sima szöveg, a sortöréseket \\n-nel írd, kódblokk és idézőjel-problémák nélkül.
- Ha sok dolgot kérnek (pl. egy teljes rangrendszert vagy szervert), minden elemet külön műveletként add vissza (pl. minden rang egy create_role). Rangoknál adj színt és értelmes jogokat, a legfontosabb rangokat tedd előre.
- Rangok sorrendje: a felhasználó által kért sorrendben hozd létre őket.

PÉLDÁK:
Bemenet: "Hozz létre egy csatornát 💬│csevego néven"
Válasz: {"reply": "Létrehozom a 💬│csevego csatornát!", "actions": [{"action": "create_channel", "name": "💬│csevego", "channel_type": "text", "category": null}]}

Bemenet: "Szia, hogy vagy?"
Válasz: {"reply": "Szia! Jól vagyok, köszönöm. Miben segíthetek a szerveren?", "actions": []}
"""


IMAGE_TOKENS = 2600  # becslés: egy 1600 px-es kép kb. ennyi tokent használ


def est_tokens(text) -> int:
    return int(len(str(text)) / CHARS_PER_TOKEN) + 1


def salvage_plain_text(e):
    """Ha a modell sima szöveggel válaszolt (nem JSON-nal), a Groq hibájából kinyeri a szöveget."""
    text = None
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            text = err.get("failed_generation")
    if not text:
        m = re.search(r"failed_generation'?\"?:\s*['\"](.*)", str(e), re.DOTALL)
        text = m.group(1) if m else None
    if not text:
        return None
    t = str(text).strip()
    if t and not t.lstrip().startswith("{"):
        return {"reply": t[:6000], "actions": []}
    return None


def ask_groq_sync(user_message: str, states: list, history: list, images=None) -> dict:
    retry_note = {
        "role": "system",
        "content": (
            "Az előző válaszod hibás JSON volt, mert túl hosszú szöveget vagy kódot írtál a reply mezőbe. "
            "Most a reply legyen EGY rövid mondat, kód és sortörés nélkül, a többit az actions végezze."
        ),
    }
    level = 0
    json_failed = False
    rate_hits = 0
    models = [VISION_MODEL, VISION_FALLBACK] if images else [MODEL]
    model_idx = 0
    use_extra = True      # reasoning-beállítások (ha a modell nem ismeri, kikapcsoljuk)
    use_json_mode = True  # JSON mód (ha képpel nem megy, kikapcsoljuk)

    for attempt in range(8):
        state = states[min(level, len(states) - 1)]
        hist = history if level == 0 else (history[-2:] if level == 1 else [])
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        full_prompt = (
            f"Szerver állapota:\n{state_json}\n\n"
            f"Felhasználó üzenete: {user_message}\n\n"
            f"Válaszolj JSON-ban. A reply rövid legyen, kód nélkül, a munkát az actions végezze."
        )

        in_tokens = (
            est_tokens(SYSTEM_PROMPT) + est_tokens(full_prompt)
            + sum(est_tokens(h["content"]) for h in hist)
            + (IMAGE_TOKENS * len(images) if images else 0) + 100
        )
        budget = EFFECTIVE_TPM - in_tokens
        if budget < 700:
            if level < 2:
                level += 1
                continue
            return {
                "reply": "Ez a kérés túl hosszú a Groq percenkénti korlátjához. Bontsd kisebb részekre, vagy várj egy percet.",
                "actions": [],
            }
        max_tokens = min(4000, budget)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if json_failed:
            messages.append(retry_note)
        messages.extend(hist)
        if images:
            content = [{"type": "text", "text": full_prompt}]
            for url in images:
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": full_prompt})

        use_model = models[model_idx]
        kwargs = {}
        if use_extra:
            if "gpt-oss" in use_model:
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            elif "qwen" in use_model:
                kwargs["extra_body"] = {"reasoning_effort": "none"}
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        raw = ""
        try:
            response = GROQ_CLIENTS[_key_state["idx"]].chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.1 if json_failed else 0.3,
                max_tokens=max_tokens,
                **kwargs,
            )
            raw = (response.choices[0].message.content or "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            if images:
                print(f"[VISION] modell={use_model} válasz={raw[:300]}")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("nem objektum")
            if not isinstance(data.get("actions"), list):
                data["actions"] = []
            data["reply"] = str(data.get("reply") or "Rendben.")[:6000]
            return data
        except json.JSONDecodeError:
            json_failed = True
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            print(f"[GROQ HIBA] modell={use_model} {msg[:300]}")
            if "model_not_found" in low or "does not exist" in low or "decommissioned" in low:
                if model_idx + 1 < len(models):
                    model_idx += 1       # tartalék modell
                else:
                    hint = "VISION_MODEL" if images else "GROQ_MODEL"
                    return {
                        "reply": f"A(z) {use_model} modell nem érhető el. Írd be a .env-be a {hint}=... sort egy aktuális modellnévvel "
                                 f"(console.groq.com/docs/models).",
                        "actions": [],
                    }
            elif ("reasoning" in low) and use_extra:
                use_extra = False        # a modell nem ismeri a reasoning beállítást
            elif images and use_json_mode and ("json" in low and "json_validate_failed" not in low):
                use_json_mode = False    # képpel nem megy a JSON mód, szövegből szedjük ki
            elif "json_validate_failed" in low or "failed to generate json" in low:
                json_failed = True
                plain = salvage_plain_text(e)
                if plain is not None:
                    return plain
            elif "413" in msg or "request too large" in low:
                if level < 2:
                    level += 1          # kisebb állapot, kevesebb előzmény
                else:
                    return {"reply": "Ez a kérés túl hosszú a Groq korlátjához. Bontsd kisebb részekre.", "actions": []}
            elif "429" in msg or "rate_limit" in low or "rate limit" in low:
                rate_hits += 1
                if len(GROQ_CLIENTS) > 1 and rate_hits % len(GROQ_CLIENTS) != 0:
                    _key_state["idx"] = (_key_state["idx"] + 1) % len(GROQ_CLIENTS)   # másik kulcs
                else:
                    time.sleep(25)       # minden kulcs limites: megvárjuk a visszatöltődést
            else:
                return {"reply": f"Hiba a Groq API hívás közben: {msg[:300]}", "actions": []}

    return {
        "reply": "Ezt most nem sikerült feldolgozni. Írd le egyszerűbben, vagy bontsd kisebb részekre a kérést.",
        "actions": [],
    }


async def ask_groq(user_message, states, history, images=None):
    # a Groq hívás blokkoló, ezért külön szálon fut, hogy a bot ne fagyjon le
    return await asyncio.to_thread(ask_groq_sync, user_message, states, history, images)


# ============================================================
# 3. AKCIÓK VÉGREHAJTÁSA
# ============================================================

def private_overwrites(guild, allowed_roles):
    ow = {guild.default_role: discord.PermissionOverwrite(view_channel=False)}
    for rn in allowed_roles or []:
        r = find_role(guild, rn)
        if r:
            ow[r] = discord.PermissionOverwrite(view_channel=True, send_messages=True, connect=True)
    ow[guild.me] = discord.PermissionOverwrite(view_channel=True, send_messages=True, manage_channels=True)
    return ow


async def execute_action(guild: discord.Guild, act: dict, origin: discord.TextChannel) -> str:
    action = act.get("action")
    reason = "Pentagon AI"

    # ---------------- CSATORNÁK ----------------
    if action == "create_channel":
        cat = None
        if act.get("category"):
            cat = find_category(guild, act["category"])
            if not cat:
                cat = await guild.create_category(act["category"])
        kwargs = {"category": cat, "reason": reason}
        if act.get("private"):
            kwargs["overwrites"] = private_overwrites(guild, act.get("allowed_roles"))
        ctype = (act.get("channel_type") or "text").lower()
        name = act["name"]
        if ctype == "voice":
            ch = await guild.create_voice_channel(name, **kwargs)
        elif ctype == "stage":
            ch = await guild.create_stage_channel(name, **kwargs)
        elif ctype == "forum":
            ch = await guild.create_forum(name, **kwargs)
        else:
            if act.get("topic"):
                kwargs["topic"] = act["topic"]
            ch = await guild.create_text_channel(name, **kwargs)
            if ctype == "announcement":
                try:
                    await ch.edit(type=discord.ChannelType.news)
                except Exception:
                    pass
        return f"✅ Csatorna létrehozva: **{ch.name}**"

    if action == "delete_channel":
        ch = find_channel(guild, act["name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['name']}**"
        n = ch.name
        await ch.delete(reason=reason)
        return f"🗑️ Csatorna törölve: **{n}**"

    if action == "rename_channel":
        ch = find_channel(guild, act["old_name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['old_name']}**"
        await ch.edit(name=act["new_name"], reason=reason)
        return f"✏️ Átnevezve: **{act['old_name']}** → **{act['new_name']}**"

    if action == "set_topic":
        ch = find_channel(guild, act["channel_name"], (discord.TextChannel, discord.ForumChannel))
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['channel_name']}**"
        await ch.edit(topic=act.get("topic", ""), reason=reason)
        return f"📝 Téma beállítva: **{ch.name}**"

    if action == "move_channel":
        ch = find_channel(guild, act["channel_name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['channel_name']}**"
        cat = find_category(guild, act.get("category")) if act.get("category") else None
        if act.get("category") and not cat:
            cat = await guild.create_category(act["category"])
        await ch.edit(category=cat, reason=reason)
        return f"📦 Áthelyezve: **{ch.name}** → **{cat.name if cat else 'nincs kategória'}**"

    if action == "set_channel_position":
        ch = find_channel(guild, act["channel_name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['channel_name']}**"
        await ch.edit(position=int(act["position"]), reason=reason)
        return f"↕️ Pozíció beállítva: **{ch.name}**"

    if action == "set_slowmode":
        ch = find_channel(guild, act["channel_name"], discord.TextChannel)
        if not ch:
            return f"❌ Nincs ilyen szöveges csatorna: **{act['channel_name']}**"
        await ch.edit(slowmode_delay=int(act.get("seconds", 0)), reason=reason)
        return f"🐢 Lassított mód: **{ch.name}** – {act.get('seconds', 0)} mp"

    if action in ("lock_channel", "unlock_channel", "hide_channel", "show_channel"):
        ch = find_channel(guild, act["channel_name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['channel_name']}**"
        ow = ch.overwrites_for(guild.default_role)
        if action == "lock_channel":
            ow.send_messages = False
            txt = f"🔒 Lezárva: **{ch.name}**"
        elif action == "unlock_channel":
            ow.send_messages = None
            txt = f"🔓 Feloldva: **{ch.name}**"
        elif action == "hide_channel":
            ow.view_channel = False
            txt = f"🙈 Elrejtve: **{ch.name}**"
        else:
            ow.view_channel = None
            txt = f"👁️ Láthatóvá téve: **{ch.name}**"
        await ch.set_permissions(guild.default_role, overwrite=ow, reason=reason)
        return txt

    if action == "set_channel_permission":
        ch = find_channel(guild, act["channel_name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['channel_name']}**"
        target = find_role(guild, act["role_name"])
        if not target:
            return f"❌ Nincs ilyen rang: **{act['role_name']}**"
        perm = act["permission"]
        if perm not in discord.Permissions.VALID_FLAGS:
            return f"❌ Ismeretlen jogosultság: **{perm}**"
        ow = ch.overwrites_for(target)
        setattr(ow, perm, act.get("allow", True))
        await ch.set_permissions(target, overwrite=ow, reason=reason)
        return f"🔒 Jogosultság frissítve: **{ch.name}** / **{target.name}** / {perm}"

    if action == "clone_channel":
        ch = find_channel(guild, act["name"])
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act['name']}**"
        new = await ch.clone(name=act.get("new_name"), reason=reason)
        return f"📑 Klónozva: **{new.name}**"

    # ---------------- KATEGÓRIÁK ----------------
    if action == "create_category":
        kwargs = {"reason": reason}
        if act.get("private"):
            kwargs["overwrites"] = private_overwrites(guild, act.get("allowed_roles"))
        cat = await guild.create_category(act["name"], **kwargs)
        return f"✅ Kategória létrehozva: **{cat.name}**"

    if action == "delete_category":
        cat = find_category(guild, act["name"])
        if not cat:
            return f"❌ Nincs ilyen kategória: **{act['name']}**"
        n = cat.name
        if act.get("delete_channels"):
            for c in list(cat.channels):
                await c.delete(reason=reason)
                await asyncio.sleep(0.3)
        await cat.delete(reason=reason)
        return f"🗑️ Kategória törölve: **{n}**"

    if action == "rename_category":
        cat = find_category(guild, act["old_name"])
        if not cat:
            return f"❌ Nincs ilyen kategória: **{act['old_name']}**"
        await cat.edit(name=act["new_name"], reason=reason)
        return f"✏️ Kategória átnevezve: **{act['old_name']}** → **{act['new_name']}**"

    # ---------------- ÜZENETEK ----------------
    if action == "send_message":
        ch = find_channel(guild, act.get("channel_name"), (discord.TextChannel, discord.Thread)) \
            if act.get("channel_name") else origin
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act.get('channel_name')}**"
        if act.get("embed"):
            emb = discord.Embed(
                title=act.get("title"),
                description=act.get("text", ""),
                colour=parse_color(act.get("color", "5865F2")),
            )
            await ch.send(embed=emb)
        else:
            for part in split_message(act.get("text", "")):
                await ch.send(part)
        return f"💬 Üzenet elküldve: **{ch.name}**"

    if action == "purge_messages":
        ch = find_channel(guild, act.get("channel_name"), discord.TextChannel) \
            if act.get("channel_name") else origin
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act.get('channel_name')}**"
        amount = max(1, min(int(act.get("amount", 10)), 500))
        deleted = await ch.purge(limit=amount)
        return f"🧹 Törölt üzenetek: **{len(deleted)}** ({ch.name})"

    if action == "create_invite":
        ch = find_channel(guild, act.get("channel_name"), (discord.TextChannel, discord.VoiceChannel)) \
            if act.get("channel_name") else origin
        if not ch:
            return f"❌ Nincs ilyen csatorna: **{act.get('channel_name')}**"
        inv = await ch.create_invite(
            max_age=int(act.get("max_age_seconds", 0)),
            max_uses=int(act.get("max_uses", 0)),
            reason=reason,
        )
        return f"🔗 Meghívó: {inv.url}"

    # ---------------- RANGOK ----------------
    if action == "create_role":
        perms = make_permissions(act.get("permissions"))
        if act.get("administrator"):
            perms.administrator = True
        if act.get("all_permissions"):
            perms = discord.Permissions.all()
        role = await guild.create_role(
            name=act["name"],
            colour=parse_color(act.get("color", "000000")),
            hoist=bool(act.get("hoist", False)),
            mentionable=bool(act.get("mentionable", False)),
            permissions=perms,
            reason=reason,
        )
        return f"✅ Rang létrehozva: **{role.name}**"

    if action == "delete_role":
        role = find_role(guild, act["name"])
        if not role or role.is_default():
            return f"❌ Nincs ilyen rang: **{act['name']}**"
        n = role.name
        await role.delete(reason=reason)
        return f"🗑️ Rang törölve: **{n}**"

    if action == "edit_role":
        role = find_role(guild, act["name"])
        if not role or role.is_default():
            return f"❌ Nincs ilyen rang: **{act['name']}**"
        kwargs = {"reason": reason}
        if act.get("new_name"):
            kwargs["name"] = act["new_name"]
        if act.get("color"):
            kwargs["colour"] = parse_color(act["color"])
        if act.get("hoist") is not None:
            kwargs["hoist"] = bool(act["hoist"])
        if act.get("mentionable") is not None:
            kwargs["mentionable"] = bool(act["mentionable"])
        if act.get("all_permissions"):
            kwargs["permissions"] = discord.Permissions.all()
        elif act.get("clear_permissions"):
            kwargs["permissions"] = discord.Permissions.none()
        elif act.get("add_permissions") or act.get("remove_permissions"):
            perms = discord.Permissions(role.permissions.value)
            for p in act.get("add_permissions") or []:
                if p in discord.Permissions.VALID_FLAGS:
                    setattr(perms, p, True)
            for p in act.get("remove_permissions") or []:
                if p in discord.Permissions.VALID_FLAGS:
                    setattr(perms, p, False)
            kwargs["permissions"] = perms
        await role.edit(**kwargs)
        if act.get("all_permissions") or act.get("clear_permissions"):
            have = sum(1 for _, v in role.permissions if v)
            total = sum(1 for _, v in discord.Permissions.all() if v)
            return f"✏️ Rang módosítva: **{role.name}** – jogosultságok: **{have}/{total}**"
        return f"✏️ Rang módosítva: **{role.name}**"

    if action == "add_role":
        member = find_member(guild, act["member_name"])
        role = find_role(guild, act["role_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        if not role:
            return f"❌ Nincs ilyen rang: **{act['role_name']}**"
        await member.add_roles(role, reason=reason)
        return f"✅ {member.display_name} megkapta: **{role.name}**"

    if action == "remove_role":
        member = find_member(guild, act["member_name"])
        role = find_role(guild, act["role_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        if not role:
            return f"❌ Nincs ilyen rang: **{act['role_name']}**"
        await member.remove_roles(role, reason=reason)
        return f"✅ {member.display_name} elvesztette: **{role.name}**"

    # ---------------- TAGOK ----------------
    if action == "ban_member":
        member = find_member(guild, act["member_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        await member.ban(reason=act.get("reason") or "Nincs indok")
        return f"🔨 Kitiltva: **{member.display_name}**"

    if action == "unban_member":
        target = str(act["user"]).strip()
        async for entry in guild.bans(limit=None):
            u = entry.user
            if str(u.id) == target or u.name.lower() == target.lower():
                await guild.unban(u, reason=act.get("reason") or reason)
                return f"♻️ Kitiltás feloldva: **{u.name}**"
        return f"❌ Nincs ilyen kitiltott felhasználó: **{target}**"

    if action == "kick_member":
        member = find_member(guild, act["member_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        await member.kick(reason=act.get("reason") or "Nincs indok")
        return f"👢 Kirúgva: **{member.display_name}**"

    if action == "timeout_member":
        member = find_member(guild, act["member_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        minutes = max(1, min(int(act.get("minutes", 10)), 60 * 24 * 28))
        await member.timeout(timedelta(minutes=minutes), reason=act.get("reason") or reason)
        return f"🤐 Némítva: **{member.display_name}** ({minutes} perc)"

    if action == "untimeout_member":
        member = find_member(guild, act["member_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        await member.timeout(None, reason=reason)
        return f"🔊 Némítás feloldva: **{member.display_name}**"

    if action == "set_nickname":
        member = find_member(guild, act["member_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        await member.edit(nick=act.get("nickname"), reason=reason)
        return f"🏷️ Becenév beállítva: **{member.name}**"

    if action == "move_member_voice":
        member = find_member(guild, act["member_name"])
        if not member:
            return f"❌ Nincs ilyen tag: **{act['member_name']}**"
        if not member.voice:
            return f"❌ **{member.display_name}** nincs hangcsatornában."
        dest = None
        if act.get("channel_name"):
            dest = find_channel(guild, act["channel_name"], (discord.VoiceChannel, discord.StageChannel))
            if not dest:
                return f"❌ Nincs ilyen hangcsatorna: **{act['channel_name']}**"
        await member.move_to(dest, reason=reason)
        return f"🎧 Áthelyezve: **{member.display_name}**"

    # ---------------- SZERVER ----------------
    if action == "rename_guild":
        await guild.edit(name=act["new_name"], reason=reason)
        return f"🏷️ Szerver átnevezve: **{act['new_name']}**"

    return f"⚠️ Ismeretlen akció: **{action}**"


TEXT_EXTS = (".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".yml", ".yaml", ".xml", ".html")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
MAX_TEXT_CHARS = 30000
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 3_000_000

def prepare_image(data: bytes, mime: str) -> str:
    """Átméretezi és JPEG-re tömöríti a képet, hogy elférjen a Groq korlátaiban."""
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data, mime = buf.getvalue(), "image/jpeg"
    except ImportError:
        print("[KÉP] A Pillow nincs telepítve (pip install pillow), eredeti képet küldöm.")
    except Exception as e:
        print(f"[KÉP] Átméretezési hiba: {e}")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


async def read_attachments(message: discord.Message):
    """Visszaadja: (a csatolt szövegfájlok tartalma, képek data-URL listája, figyelmeztetések)."""
    texts, images, notes = [], [], []
    for att in message.attachments:
        name = att.filename.lower()
        ctype = (att.content_type or "").lower()
        try:
            if ctype.startswith("image/") or name.endswith(IMAGE_EXTS):
                if len(images) >= MAX_IMAGES:
                    notes.append(f"⚠️ Csak az első {MAX_IMAGES} képet olvasom el.")
                    continue
                if att.size > 25_000_000:
                    notes.append(f"⚠️ A(z) {att.filename} túl nagy kép (max 25 MB).")
                    continue
                data = await att.read()
                mime = ctype.split(";")[0] if ctype.startswith("image/") else "image/png"
                url = prepare_image(data, mime)
                print(f"[KÉP] {att.filename}: eredeti {len(data)//1024} KB -> küldött {len(url)//1024} KB (base64)")
                if len(url) > 5_200_000:
                    notes.append(f"⚠️ A(z) {att.filename} még tömörítve is túl nagy a Groq-nak. Küldj kisebb képet.")
                    continue
                images.append(url)
            elif ctype.startswith("text/") or name.endswith(TEXT_EXTS):
                if att.size > 300_000:
                    notes.append(f"⚠️ A(z) {att.filename} túl nagy szövegfájl (max 300 KB).")
                    continue
                data = await att.read()
                text = data.decode("utf-8", errors="replace")
                if len(text) > MAX_TEXT_CHARS:
                    notes.append(f"⚠️ A(z) {att.filename} túl hosszú, csak az első {MAX_TEXT_CHARS} karaktert olvastam el.")
                    text = text[:MAX_TEXT_CHARS]
                texts.append(f"[Csatolt fájl: {att.filename}]\n{text}")
            else:
                notes.append(f"⚠️ A(z) {att.filename} fájltípust nem tudom elolvasni (kép és szövegfájl jó).")
        except Exception as e:
            notes.append(f"⚠️ Nem sikerült beolvasni: {att.filename} ({e})")
    return texts, images, notes


# Veszélyes műveletet csak akkor hajt végre, ha a kérésben tényleg szerepel a megfelelő szó.
# (Védelem arra, ha az AI félreérti a kérést és magától törölne vagy kitiltana.)
DANGEROUS = {
    "purge_messages": r"torl|torol|tisztit|purge|seper|clear|delete|urits",
    "delete_channel": r"torl|torol|delete|szuntet|tavolit|remove",
    "delete_category": r"torl|torol|delete|szuntet|tavolit|remove",
    "delete_role": r"torl|torol|delete|szuntet|tavolit|remove",
    "ban_member": r"ban|kitilt",
    "kick_member": r"kirug|kick|rugd|rugj|rugni",
}


async def execute_actions(guild: discord.Guild, actions: list, origin: discord.TextChannel, source_text: str = ""):
    results = []
    for act in actions:
        if not isinstance(act, dict):
            continue
        action = act.get("action", "?")
        pattern = DANGEROUS.get(action)
        if pattern and not re.search(pattern, strip_accents(source_text)):
            results.append(f"🛑 A(z) **{action}** műveletet kihagytam, mert nem kérted kifejezetten.")
            continue
        try:
            results.append(await execute_action(guild, act, origin))
        except discord.Forbidden:
            results.append(
                f"❌ Nincs jogosultságom a(z) **{action}** művelethez. "
                f"Tedd a bot rangját a lista tetejére, és adj neki Administrator jogot!"
            )
        except KeyError as e:
            results.append(f"❌ Hiányzó adat a(z) **{action}** műveletnél: {e}")
        except discord.HTTPException as e:
            results.append(f"❌ Discord hiba a(z) **{action}** közben: {e.text}")
        except Exception as e:
            results.append(f"❌ Hiba a(z) **{action}** közben: {e}")
        await asyncio.sleep(0.4)  # rate limit elkerülése nagy építkezéseknél
    return results


# ============================================================
# 4. ÜZENET FIGYELŐ
# ============================================================

def can_control(member: discord.Member) -> bool:
    return (
        member.id in ALLOWED_USER_IDS
        or member.guild_permissions.administrator
        or member.id == member.guild.owner_id
    )


def split_into_chunks(text: str, max_chars: int):
    """Nagy szöveg darabolása sor/fejezethatáron."""
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > max_chars:                       # extra hosszú sor
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:max_chars])
            line = line[max_chars:]
        is_heading = line.lstrip().startswith("#")
        if cur and (len(cur) + len(line) + 1 > max_chars or (is_heading and len(cur) > max_chars * 0.4)):
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or message.guild is None:
        return

    mentioned = bot.user in message.mentions
    starts_with_name = message.content.lower().lstrip().startswith("pentagon")

    # Adminok (és az engedélyezettek) sima üzenetére is válaszol, említés nélkül.
    # Ha az AI_CHANNEL_IDS meg van adva a .env-ben, akkor csak ezekben a csatornákban.
    plain_ok = can_control(message.author) and (
        not AI_CHANNEL_IDS or message.channel.id in AI_CHANNEL_IDS
    )

    if not (mentioned or starts_with_name or plain_ok):
        return
    if not message.content.strip() and not message.attachments:
        return

    content = message.content
    content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
    content = re.sub(r"^\s*pentagon(\s+ai)?[\s,:!\-]*", "", content, flags=re.IGNORECASE).strip()
    content = resolve_mentions(message.guild, content)

    # Csatolt fájlok: szövegfájl tartalma + képek beolvasása
    texts, images, notes = [], [], []
    if message.attachments and can_control(message.author):
        async with message.channel.typing():
            texts, images, notes = await read_attachments(message)
    if notes:
        await message.channel.send("\n".join(notes))
    if message.attachments and can_control(message.author):
        print(f"[CSATOLMÁNY] {len(message.attachments)} db: " +
              ", ".join(f"{a.filename} ({a.content_type})" for a in message.attachments) +
              f" -> képek: {len(images)}, szövegfájlok: {len(texts)}")
        if images:
            await message.channel.send(f"🖼️ {len(images)} kép beolvasva, elemzem…")

    if not content and (texts or images):
        content = "Hajtsd végre a csatolt fájlban / képen lévő utasításokat."

    if not content:
        await message.reply("Szia! Miben segíthetek? Írd le, mit szeretnél a szerveren.")
        return

    full_content = content
    if texts:
        full_content += "\n\n" + "\n\n".join(texts)
    if images:
        full_content += "\n\n[A kérés mellé kép(ek) is csatolva vannak, olvasd el és hajtsd végre a rajtuk lévő utasításokat.]"

    is_admin = can_control(message.author)
    hist = HISTORY.setdefault(message.channel.id, [])

    suffix = ""
    if not is_admin:
        suffix = (
            "\n\n[Rendszerjegyzet: ez a felhasználó NEM adminisztrátor. "
            "Csak beszélgess vele, az actions legyen üres, és ha műveletet kér, "
            "mondd meg, hogy ehhez adminisztrátori jog kell.]"
        )

    # Mekkora kérés fér bele a Groq percenkénti token-limitjébe? Ha több, részletekben dolgozzuk fel.
    base_state = json.dumps(get_guild_state(message.guild, 1), ensure_ascii=False, separators=(",", ":"))
    overhead = est_tokens(SYSTEM_PROMPT) + est_tokens(base_state) + 500 + IMAGE_TOKENS * len(images)
    budget_chars = max(1500, int((EFFECTIVE_TPM - overhead - 2200) * CHARS_PER_TOKEN))
    chunks = split_into_chunks(full_content, budget_chars) if is_admin else [full_content[:budget_chars]]
    total = len(chunks)

    last_reply = ""
    for idx, chunk in enumerate(chunks, 1):
        if total > 1:
            await message.channel.send(f"⏳ Nagy kérés, részletekben dolgozom: **{idx}/{total}**")
            prompt = (
                chunk
                + f"\n\n[Ez a kérés {idx}/{total}. része. Csak az ebben a részben szereplő dolgokat hajtsd végre. "
                  f"Ami már létezik a szerveren, azt ne hozd létre újra.]"
            )
        else:
            prompt = chunk + suffix

        async with message.channel.typing():
            states = [get_guild_state(message.guild, lvl) for lvl in (0, 1, 2)]
            result = await ask_groq(prompt, states, hist, images if idx == 1 else None)

        reply_text = str(result.get("reply") or "Nem értettem.")
        actions = result.get("actions", []) if is_admin else []
        last_reply = reply_text

        try:
            for pi, part in enumerate(split_message(reply_text)):
                if idx == 1 and pi == 0:
                    await message.reply(part)
                else:
                    await message.channel.send(part)
        except discord.HTTPException:
            pass

        if actions:
            async with message.channel.typing():
                results = await execute_actions(message.guild, actions, message.channel, full_content)
            for part in split_message("\n".join(results)):
                try:
                    await message.channel.send(part)
                except discord.HTTPException:
                    break  # pl. ha magát a csatornát törölte

        if idx < total:
            await message.channel.send("⏱️ Várok kicsit a Groq percenkénti korlátja miatt…")
            await asyncio.sleep(45)

    hist.append({"role": "user", "content": content[:1500] + (" [fájl/kép csatolva]" if (texts or images) else "")})
    hist.append({"role": "assistant", "content": last_reply})
    del hist[:-HISTORY_LIMIT]
    if total > 1:
        await message.channel.send("✅ Kész, minden részt feldolgoztam.")


# ============================================================
# 5. INDÍTÁS
# ============================================================

@bot.event
async def on_ready():
    # Minden régi slash parancs törlése a Discordból (a bot csak sima üzenetekre dolgozik)
    try:
        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()
        for g in bot.guilds:
            bot.tree.clear_commands(guild=g)
            await bot.tree.sync(guild=g)
        print("Régi slash parancsok törölve.")
    except Exception as e:
        print(f"Slash törlési hiba: {e}")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="Pentagon AI"))
    print(f"Bejelentkezve: {bot.user}")
    print(f"Bot ID: {bot.user.id}")
    print(f"Szerverek: {len(bot.guilds)}")
    for g in bot.guilds:
        print(f"  - {g.name} (ID: {g.id})")


bot.run(DISCORD_TOKEN)